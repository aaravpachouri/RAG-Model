"""Generate a small sample notes PDF in data/documents for demo/testing.

The text is original sample study notes written for this project, so the
generated file is safe to commit, share, and demo.

Run from the project root:
    python scripts/create_sample_pdf.py
"""

from pathlib import Path

import pymupdf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "documents"
OUTPUT_PATH = OUTPUT_DIR / "sample_physics_notes.pdf"

PAGES = [
    (
        "Rotational Motion - Part 1\n\n"
        "Moment of inertia\n\n"
        "The moment of inertia of a point mass m at a "
        "perpendicular distance r from the axis of "
        "rotation is I = m r^2.\n\n"
        "For a system of particles, the moment of "
        "inertia is the sum I = sum of m_i r_i^2 over "
        "all particles. It depends on the mass "
        "distribution relative to the axis of "
        "rotation, not on the total mass alone.\n\n"
        "For a uniform solid sphere rotating about a "
        "diameter, I = (2/5) M R^2. For a uniform "
        "thin rod about its centre, I = (1/12) M L^2."
    ),
    (
        "Rotational Motion - Part 2\n\n"
        "Torque\n\n"
        "Torque is the rotational analogue of force. "
        "For a force F applied at a perpendicular "
        "distance r from the pivot, the magnitude of "
        "the torque is tau = r F sin(theta), where "
        "theta is the angle between r and F.\n\n"
        "The SI unit of torque is the newton metre "
        "(N m). A net torque changes the angular "
        "momentum of a body: tau = dL/dt."
    ),
    (
        "Rotational Motion - Part 3\n\n"
        "Angular momentum\n\n"
        "A particle with linear momentum p at "
        "perpendicular distance r from the axis has "
        "angular momentum L = r p. A rigid body "
        "rotating with angular velocity omega has "
        "L = I omega.\n\n"
        "Conservation: when no external torque acts "
        "on a system, its total angular momentum "
        "stays constant. This is why a figure skater "
        "spins faster after pulling in their arms: "
        "the moment of inertia decreases, so the "
        "angular velocity increases."
    ),
]


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    document = pymupdf.open()

    for text in PAGES:

        page = document.new_page()

        page.insert_text(
            pymupdf.Point(72, 72),
            text,
            fontsize=11,
        )

    document.save(str(OUTPUT_PATH))
    document.close()

    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
