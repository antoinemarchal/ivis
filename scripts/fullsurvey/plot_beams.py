import os
from astropy.io import fits
import matplotlib.pyplot as plt

root = "/totoro/anmarchal/data/gaskap/fullsurvey/holography_beams/79298"

ra_list = []
dec_list = []
names = []

# --- Walk the directory and find all FITS files ---
for dirpath, dirnames, filenames in os.walk(root):
    for fname in filenames:
        if fname.lower().endswith(".fits"):
            fpath = os.path.join(dirpath, fname)

            try:
                with fits.open(fpath) as hdul:
                    hdr = hdul[0].header

                    # Try standard WCS keywords
                    ra = hdr.get("CRVAL1")
                    dec = hdr.get("CRVAL2")

                    if ra is None or dec is None:
                        print(f"[WARNING] No CRVAL1/CRVAL2 in {fpath}")
                        continue

                    ra_list.append(ra)
                    dec_list.append(dec)
                    names.append(fname)

                    print(f"Found RA={ra:.6f}, DEC={dec:.6f} in {fname}")

            except Exception as e:
                print(f"[ERROR] Could not read {fpath}: {e}")


# --- Plot ---
plt.figure(figsize=(8, 6))
plt.scatter(ra_list, dec_list, s=40)

# Add labels (optional)
for x, y, name in zip(ra_list, dec_list, names):
    plt.text(x, y, name.replace(".fits",""), fontsize=7)

plt.xlabel("RA (deg)")
plt.ylabel("DEC (deg)")
plt.title("Beam Center Positions from FITS Headers")
plt.grid(True)
plt.tight_layout()

# --- Save to file ---
output_file = "beam_centers.png"
plt.savefig(output_file, dpi=200)

print(f"Saved figure to {output_file}")
