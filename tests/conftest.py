import sys
from pathlib import Path

import pymupdf
import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT / "src"))


SAMPLE_PAGES = [
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


@pytest.fixture(scope="module")
def sample_pdf(tmp_path_factory):
    """A small generated notes PDF with known content."""

    path = tmp_path_factory.mktemp("pdfs") / "sample_notes.pdf"

    document = pymupdf.open()

    for text in SAMPLE_PAGES:
        page = document.new_page()
        page.insert_text(
            pymupdf.Point(72, 72),
            text,
            fontsize=11,
        )

    document.save(str(path))
    document.close()

    return path
