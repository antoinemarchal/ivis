from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil

SPW_MAP = {
    "M33_A": "5:2000~2001",
    "M33_B": "8:2000~2001",
    "M33_C": "4:2000~2001",
    "M33_D": "4:2000~2001",
}

INPUT_DIR = Path("/totoro/anmarchal/data/lglbs/msets")
OUTPUT_DIR = Path("/totoro/anmarchal/data/lglbs/pb")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_spw(ms_name: str) -> str:
    for key, spw in SPW_MAP.items():
        if key in ms_name:
            return spw
    raise ValueError(f"Unknown dataset: {ms_name}")


def process_one(ms_path_str: str) -> str:
    from casatasks import tclean, exportfits

    ms = Path(ms_path_str)
    spw = get_spw(ms.name)
    imagename = OUTPUT_DIR / ms.stem

    pb_fits = Path(str(imagename) + ".pb.fits")
    if pb_fits.exists():
        return f"SKIP existing: {pb_fits.name}"

    # optional: remove incomplete previous CASA products
    for suffix in [".image", ".image.pbcor", ".model", ".pb", ".psf", ".residual", ".sumwt"]:
        p = Path(str(imagename) + suffix)
        if p.exists():
            shutil.rmtree(p)

    tclean(
        vis=str(ms),
        spw=spw,
        imagename=str(imagename),
        restfreq="1.42040571183GHz",
        uvrange="",
        imsize=7200,
        specmode="cube",
        deconvolver="hogbom",
        threshold="1.4mJy",
        gridder="standard",
        weighting="natural",
        cell="0.75arcsec",
        niter=0,
        calcpsf=True,
        calcres=True,
        pbcor=True,
        interactive=False,
        usemask="pb",
    )

    exportfits(
        imagename=str(imagename) + ".pb",
        fitsimage=str(imagename) + ".pb.fits",
        dropstokes=True,
        overwrite=True,
    )

    return f"DONE: {ms.name} -> {pb_fits.name}"


def main():
    ms_list = sorted(INPUT_DIR.glob("*.ms"))

    # Start conservative. tclean is heavy in RAM and I/O.
    n_workers = 12

    print(f"Found {len(ms_list)} MS")
    print(f"Running with {n_workers} workers")

    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        futures = {exe.submit(process_one, str(ms)): ms for ms in ms_list}

        for fut in as_completed(futures):
            ms = futures[fut]
            try:
                print(fut.result(), flush=True)
            except Exception as e:
                print(f"FAILED: {ms.name}", flush=True)
                print(e, flush=True)


if __name__ == "__main__":
    main()




# stop


# from pathlib import Path
# from casatasks import tclean, exportfits

# path = Path("/totoro/anmarchal/data/lglbs/msets")
# output_dir = Path("/totoro/anmarchal/data/lglbs/pb")
# output_dir.mkdir(parents=True, exist_ok=True)

# # HI SPW for each dataset
# spw_map = {
#     "M33_A": "5:2000~2001",
#     "M33_B": "8:2000~2001",
#     "M33_C": "4:2000~2001",
#     "M33_D": "4:2000~2001",
# }

# for ms in sorted(path.glob("*.ms")):
#     # Determine which SPW to use
#     spw = None
#     for key, value in spw_map.items():
#         if key in ms.name:
#             spw = value
#             break

#     if spw is None:
#         print(f"Skipping {ms.name}: unknown dataset")
#         continue

#     imagename = output_dir / ms.stem

#     print(f"\nProcessing {ms.name}")
#     print(f"Using SPW {spw}")

#     tclean(
#         vis=str(ms),
#         spw=spw,
#         imagename=str(imagename),
#         restfreq="1.42040571183GHz",
#         uvrange="",
#         imsize=int(7200),
#         specmode="cube",
#         deconvolver="hogbom",
#         threshold="1.4mJy",
#         gridder="standard",
#         weighting="natural",
#         cell="0.75arcsec",
#         niter=0,
#         calcpsf=True,
#         calcres=True,
#         pbcor=True,
#         interactive=False,
#         usemask="pb",
#     )

#     exportfits(
#         imagename=str(imagename) + ".pb",
#         fitsimage=str(imagename) + ".pb.fits",
#         dropstokes=True,
#         overwrite=True,
#     )

#     print(f"Created {imagename}.pb")
#     print(f"Created {imagename}.pb.fits")

# print("\nDone.")


# # stop

# from pathlib import Path
# from casatasks import tclean, exportfits

# path = Path("/totoro/anmarchal/data/lglbs/msets")
# output_dir = Path("/totoro/anmarchal/data/lglbs/pb")
# output_dir.mkdir(parents=True, exist_ok=True)

# ms = path / "M33_D_20A-346.sb42432794.eb42529192.59808.23489403936.speclines.ms.split_M33_8.ms"

# imagename = output_dir / ms.stem

# tclean(
#     vis=str(ms),
#     spw="4:2000~2001",
#     imagename=str(imagename),
#     restfreq="1.42040571183GHz",
#     uvrange="",
#     imsize=int(7200)
#     specmode="cube",
#     deconvolver="hogbom",
#     threshold="1.4mJy",
#     gridder="standard",
#     weighting="natural",
#     cell="0.75arcsec",
#     niter=0,
#     calcpsf=True,
#     calcres=True,
#     pbcor=True,
#     interactive=False,
#     usemask="pb",
# )

# exportfits(
#     imagename=str(imagename) + ".pb",
#     fitsimage=str(imagename) + ".pb.fits",
#     dropstokes=True,
#     overwrite=True,
# )

# print(f"Created {imagename}.pb")
# print(f"Created {imagename}.pb.fits")

