"""This can be used to recreate the available pressure level tables if more are added"""
from ufs2arco.sources.rda_gfs_archive import RDAGFSArchive
from ufs2arco.sources.aws_hrrr_archive import AWSHRRRArchive
from ufs2arco.sources.aws_gefs_archive import AWSGEFSArchive

for name, Archive in zip(
    ["gfs", "gefs", "hrrr"],
    [RDAGFSArchive, AWSGEFSArchive, AWSHRRRArchive],
):

    kw = {}
    if name == "gefs":
        kw["member"]={"start": 0, "end": 1, "step": 1}


    levels = Archive(
        t0={"start": "2000-01-01", "end": "2000-01-02", "freq" :"6h"},
        fhr={"start": 0, "end": 6, "step": 6},
        **kw
    ).available_levels

    # Prepare rows with 3 columns each, padding as needed
    num_cols = 5
    rows = [levels[i:i+num_cols] for i in range(0, len(levels), num_cols)]

    with open(f"levels.{name}.rst", "w") as out:
        out.write(".. list-table:: Available Pressure Levels (hPa)\n")
        out.write(f"   :widths: {' '.join(['6']*num_cols)}\n")
        out.write("   :header-rows: 0\n\n")
        for row in rows:
            # Pad row to always have num_cols columns
            row_padded = tuple(row) + ('',) * (num_cols - len(row))
            out.write("   * - " + "\n     - ".join(str(cell) for cell in row_padded) + "\n")
