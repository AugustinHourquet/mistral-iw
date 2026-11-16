"""Agent tools. Each tool registers its JSON schema with the `@tool` decorator.

Tools receive the shared `httpx.AsyncClient` as their first argument (so tests can
swap in a mock transport); the model only sees the other parameters.
Errors are never raised to the agent: `execute_tool` turns them into `{"error": ...}`.
"""

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
USER_AGENT = "mistral-iw-weather-agent/0.1 (demo app)"

ToolFn = Callable[..., Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: ToolFn

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


TOOLS: dict[str, Tool] = {}


def tool(description: str, parameters: dict[str, Any]) -> Callable[[ToolFn], ToolFn]:
    """Register an async function as a tool the model can call."""

    def register(fn: ToolFn) -> ToolFn:
        TOOLS[fn.__name__] = Tool(fn.__name__, description, parameters, fn)
        return fn

    return register


def tool_schemas() -> list[dict[str, Any]]:
    return [t.schema for t in TOOLS.values()]


async def execute_tool(client: httpx.AsyncClient, name: str, arguments: str | dict) -> dict[str, Any]:
    """Run a tool by name. Always returns a dict; failures become `{"error": ...}`."""
    if name not in TOOLS:
        return {"error": f"Unknown tool '{name}'. Available tools: {', '.join(TOOLS)}."}
    try:
        args = json.loads(arguments or "{}") if isinstance(arguments, str) else dict(arguments)
        if not isinstance(args, dict):
            raise TypeError("arguments must be a JSON object")
    except (ValueError, TypeError) as exc:
        return {"error": f"Invalid arguments for {name}: {exc}"}
    try:
        return await TOOLS[name].fn(client, **args)
    except TypeError as exc:  # missing / unexpected arguments
        return {"error": f"Invalid arguments for {name}: {exc}"}
    except httpx.HTTPStatusError as exc:
        return {"error": f"{name} failed: HTTP {exc.response.status_code} from {exc.request.url.host}"}
    except httpx.HTTPError as exc:
        return {"error": f"{name} failed: {type(exc).__name__}: {exc}"}
    except Exception as exc:  # noqa: BLE001 - tool errors go back to the model
        return {"error": f"{name} failed: {type(exc).__name__}: {exc}"}


# --- WMO weather codes (https://open-meteo.com/en/docs) ---------------------------

WMO_CODES: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snowfall",
    73: "moderate snowfall",
    75: "heavy snowfall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


def describe_weather_code(code: int | None) -> str:
    if code is None:
        return "unknown"
    return WMO_CODES.get(int(code), f"unknown (WMO code {code})")


# --- Tools --------------------------------------------------------------------------


@tool(
    description=(
        "Find the coordinates of a place (city, address, landmark). "
        "Always call this before get_forecast."
    ),
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "Place name, ideally with country, e.g. 'Paris, France'.",
            }
        },
        "required": ["location"],
    },
)
async def geocode(client: httpx.AsyncClient, location: str) -> dict[str, Any]:
    location = (location or "").strip()
    if not location:
        return {"error": "location is empty"}
    resp = await client.get(
        NOMINATIM_URL,
        params={"q": location, "format": "jsonv2", "limit": 1, "addressdetails": 1},
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en"},
        timeout=10,
    )
    resp.raise_for_status()
    results = resp.json()
    if not results:
        return {"error": f"No place found for '{location}'."}
    top = results[0]
    address = top.get("address") or {}
    return {
        "name": top.get("display_name", location),
        "latitude": round(float(top["lat"]), 4),
        "longitude": round(float(top["lon"]), 4),
        "country": address.get("country"),
        "country_code": (address.get("country_code") or "").upper() or None,
    }


@tool(
    description=(
        "Get current conditions and a daily forecast for coordinates returned by geocode. "
        "Dates are local to the location; daily[0] is today there."
    ),
    parameters={
        "type": "object",
        "properties": {
            "latitude": {"type": "number", "description": "Latitude in decimal degrees."},
            "longitude": {"type": "number", "description": "Longitude in decimal degrees."},
            "days": {
                "type": "integer",
                "description": "Number of forecast days, 1-16. Default 7.",
                "minimum": 1,
                "maximum": 16,
            },
            "temperature_unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit. Default celsius.",
            },
        },
        "required": ["latitude", "longitude"],
    },
)
async def get_forecast(
    client: httpx.AsyncClient,
    latitude: float,
    longitude: float,
    days: int = 7,
    temperature_unit: str = "celsius",
) -> dict[str, Any]:
    latitude, longitude = float(latitude), float(longitude)
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return {"error": f"Invalid coordinates ({latitude}, {longitude})."}
    if temperature_unit not in ("celsius", "fahrenheit"):
        return {"error": "temperature_unit must be 'celsius' or 'fahrenheit'."}
    days = max(1, min(16, int(days)))

    resp = await client.get(
        OPEN_METEO_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "timezone": "auto",
            "forecast_days": days,
            "temperature_unit": temperature_unit,
            "wind_speed_unit": "kmh",
            "current": ",".join([
                "temperature_2m",
                "apparent_temperature",
                "relative_humidity_2m",
                "precipitation",
                "weather_code",
                "wind_speed_10m",
                "wind_gusts_10m",
            ]),
            "daily": ",".join([
                "weather_code",
                "temperature_2m_min",
                "temperature_2m_max",
                "precipitation_probability_max",
                "precipitation_sum",
                "wind_speed_10m_max",
            ]),
        },
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    current = data.get("current") or {}
    daily = data.get("daily") or {}
    t = "°F" if temperature_unit == "fahrenheit" else "°C"

    def col(key: str) -> list:
        return daily.get(key) or []

    return {
        "timezone": data.get("timezone"),
        "units": {"temperature": t, "precipitation": "mm", "wind": "km/h", "humidity": "%"},
        "current": {
            "time": current.get("time"),
            "temperature": current.get("temperature_2m"),
            "feels_like": current.get("apparent_temperature"),
            "humidity": current.get("relative_humidity_2m"),
            "precipitation": current.get("precipitation"),
            "wind_speed": current.get("wind_speed_10m"),
            "wind_gusts": current.get("wind_gusts_10m"),
            "conditions": describe_weather_code(current.get("weather_code")),
        },
        "daily": [
            {
                "date": d,
                "conditions": describe_weather_code(_at(col("weather_code"), i)),
                "temp_min": _at(col("temperature_2m_min"), i),
                "temp_max": _at(col("temperature_2m_max"), i),
                "precip_probability_max": _at(col("precipitation_probability_max"), i),
                "precip_sum": _at(col("precipitation_sum"), i),
                "wind_max": _at(col("wind_speed_10m_max"), i),
            }
            for i, d in enumerate(col("time"))
        ],
    }


def _at(values: list, i: int) -> Any:
    return values[i] if i < len(values) else None
