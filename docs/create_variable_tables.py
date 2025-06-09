"""This can be used to recreate the available variables tables if more variables are added"""
import yaml

for forecast_system in ["gefs", "gfs", "hrrr"]:

    with open(f"../ufs2arco/sources/reference.{forecast_system}.yaml") as f:
        data = yaml.safe_load(f)

    with open(f"variables.{forecast_system}.rst", "w") as out:
        # Write the table header
        out.write(f".. list-table:: Variables from {forecast_system.upper()} available in ufs2arco\n")
        if forecast_system == "gfs":
            out.write("   :widths: 18 50 12\n")
        else:
            out.write("   :widths: 18 50\n")
        out.write("   :header-rows: 1\n\n")
        out.write("   * - Variable\n")
        out.write("     - Long Name\n")

        if forecast_system == "gfs":
            out.write("     - Forecast Only\n")
        for var, attrs in data.items():
            long_name = attrs.get("long_name", "")
            out.write(f"   * - ``{var}``\n")
            out.write(f"     - {long_name}\n")

            if forecast_system == "gfs":
                forecast_only = str(attrs.get("forecast_only", None))
                out.write(f"     - {forecast_only}\n")
