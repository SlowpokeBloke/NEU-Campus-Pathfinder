
import requests

#using open meteo for current weather
def get_weather():
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": 42.3398,
        "longitude": -71.0892,
        "current": "temperature,precipitation,rain,showers,snowfall,weather_code",
        "timezone": "America/New_York"
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()

        current = response.json()["current"]

        code = current["weather_code"]

        snow_codes = [71, 73, 75, 77, 85, 86]
        rain_codes = [
            51, 53, 55, 56, 57,
            61, 63, 65, 66, 67,
            80, 81, 82, 95, 96, 99
        ]

        if code in snow_codes or current["snowfall"] > 0:
            condition = "Snow"
        elif code in rain_codes or current["rain"] > 0 or current["showers"] > 0:
            condition = "Rain"
        else:
            condition = "Normal"

        return {
            "condition": condition,
            "temperature": current["temperature"],
            "precipitation": current["precipitation"]
        }

    except requests.exceptions.RequestException as e:
        print("Weather API error:", e)
        return None


# Get current weather
weather = get_weather()

if weather:
    print("Weather:", weather["condition"])
    print("Temperature:", weather["temperature"], "°C")
    print("Precipitation:", weather["precipitation"], "mm")
