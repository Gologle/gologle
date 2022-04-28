from __future__ import annotations

import pytest

from src.parsers.cranfield import CranfieldEntry, CranfieldParser


@pytest.mark.parametrize(
    "text,expected",
    [
        # 1rst test case
        ("""41
.T
on transition experiments at moderate supersonic speeds .
.A
morkovin,m.v.
.B
j. ae. scs. 24, 1957, 480.
.W
on transition experiments at moderate supersonic speeds .
  studies of transition over a flat plate at mach number 1.76
were carried out using a hot-wire anemometer as one of the
principal tools .  the nature and measurements of free-stream
disturbances at supersonic speeds are analyzed .  the
experimental results are interpreted in the light of present overall
information on transition at supersonic speeds and conclusions as
to further fruitful experiments are drawn .""",
        {
            "id": 41,
            "title": "on transition experiments at moderate supersonic speeds .",
            "author": "morkovin,m.v.",
            "B": "j. ae. scs. 24, 1957, 480.",
            "text": """on transition experiments at moderate supersonic speeds .
  studies of transition over a flat plate at mach number 1.76
were carried out using a hot-wire anemometer as one of the
principal tools .  the nature and measurements of free-stream
disturbances at supersonic speeds are analyzed .  the
experimental results are interpreted in the light of present overall
information on transition at supersonic speeds and conclusions as
to further fruitful experiments are drawn .""",
        }),
        # 2dn test case
        ("""55
.T
separation, stability and other properties of compressible
laminar boundary layer with pressure gradient and heat
transfer .
.A
morduchow,m. and grape,r.g.
.B
naca tn.3296, 1955.
.W
separation, stability and other properties of compressible
laminar boundary layer with pressure gradient and heat
transfer .
  a theoretical study is made of the effect of pressure gradient,
wall temperature, and mach number on laminar boundary-layer
characteristics and, in particular, on the skin-friction and heat-transfer
coefficients, on the separation point in an adverse pressure gradient,
on the wall temperature required for complete stabilization of the
laminar boundary layer, and on the minimum critical reynolds number for
laminar stability .  the prandtl number is assumed to be unity and the
coefficient of viscosity is assumed to be proportional to the
temperature, with a factor arising from the sutherland relation .  a simple and
accurate method of locating the separation point in a compressible flow
with heat transfer is developed .  numerical examples to illustrate the
results in detail are given throughout .""",
         {
             "id": 55,
             "title": """separation, stability and other properties of compressible
laminar boundary layer with pressure gradient and heat
transfer .""",
             "author": "morduchow,m. and grape,r.g.",
             "B": "naca tn.3296, 1955.",
             "text": """separation, stability and other properties of compressible
laminar boundary layer with pressure gradient and heat
transfer .
  a theoretical study is made of the effect of pressure gradient,
wall temperature, and mach number on laminar boundary-layer
characteristics and, in particular, on the skin-friction and heat-transfer
coefficients, on the separation point in an adverse pressure gradient,
on the wall temperature required for complete stabilization of the
laminar boundary layer, and on the minimum critical reynolds number for
laminar stability .  the prandtl number is assumed to be unity and the
coefficient of viscosity is assumed to be proportional to the
temperature, with a factor arising from the sutherland relation .  a simple and
accurate method of locating the separation point in a compressible flow
with heat transfer is developed .  numerical examples to illustrate the
results in detail are given throughout ."""
         }),
        # 3rd test case
        ("""65
.T
convection of a pattern of vorticity through a shock
wave .
.A
ribner,h.s.
.B
naca tn.2864, 1953.
.W
convection of a pattern of vorticity through a shock
wave .
  an arbitrary weak spatial distribution of vorticity can be
represented in terms of plane sinusoidal shear waves of all orientations and
wave lengths (fourier integral) .  the analysis treats the passage of a
single representative weak shear wave through a plane shock and shows
refraction and modification of the shear wave with simultaneous
generation of an acoustically intense sound
wave .  applications to turbulence
and to noise in supersonic wind tunnels are indicated .""",
         {
             "id": 65,
             "title": """convection of a pattern of vorticity through a shock
wave .""",
             "author": "ribner,h.s.",
             "B": "naca tn.2864, 1953.",
             "text": """convection of a pattern of vorticity through a shock
wave .
  an arbitrary weak spatial distribution of vorticity can be
represented in terms of plane sinusoidal shear waves of all orientations and
wave lengths (fourier integral) .  the analysis treats the passage of a
single representative weak shear wave through a plane shock and shows
refraction and modification of the shear wave with simultaneous
generation of an acoustically intense sound
wave .  applications to turbulence
and to noise in supersonic wind tunnels are indicated ."""
         }),
        # 4th test case
        (
            """71
.T
laminar boundary layer behind shock advancing into
stationary fluid .
.A
mirels,h.
.B
naca tn.3401, 1955.
.W
laminar boundary layer behind shock advancing into
stationary fluid .
  a study was made of the laminar compressible boundary layer induced
by a shock wave advancing into a stationary fluid bounded by a wall .
for weak shock waves, the boundary layer is identical with that which
occurs when an infinite wall is impulsively set into uniform motion
shocks .
  velocity and temperature profiles, recovery factors, and
skin-friction and heat-transfer coefficients are tabulated for a wide range
of shock strengths .""",
            {
                "id": 71,
                "title": """laminar boundary layer behind shock advancing into
stationary fluid .""",
                "author": "mirels,h.",
                "B": "naca tn.3401, 1955.",
                "text": """laminar boundary layer behind shock advancing into
stationary fluid .
  a study was made of the laminar compressible boundary layer induced
by a shock wave advancing into a stationary fluid bounded by a wall .
for weak shock waves, the boundary layer is identical with that which
occurs when an infinite wall is impulsively set into uniform motion
shocks .
  velocity and temperature profiles, recovery factors, and
skin-friction and heat-transfer coefficients are tabulated for a wide range
of shock strengths ."""
            }
        ),
        # 5th test case
        (
            """74
.T
an experimental study of the turbulen coundary layer
on a shock tube wall .
.A
gooderum,p.n.
.B
naca tn.4243, 1958.
.W
an experimental study of the turbulen coundary layer
on a shock tube wall .
  interferometric measurements were made of the density profiles of
an unsteady turbulent boundary layer on the flat wall of a shock tube .
the investigation included both subsonic and supersonic flow (mach
numbers of 0.50 and 1.77) with no pressure gradient and with heat transfer
to a cold wall .  velocity profiles and average skin-friction
coefficients were calculated .  effects on the velocity profile of
surface roughness and flow length are examined .""",
            {
                "id": 74,
                "title": """an experimental study of the turbulen coundary layer
on a shock tube wall .""",
                "author": "gooderum,p.n.",
                "B": "naca tn.4243, 1958.",
                "text": """an experimental study of the turbulen coundary layer
on a shock tube wall .
  interferometric measurements were made of the density profiles of
an unsteady turbulent boundary layer on the flat wall of a shock tube .
the investigation included both subsonic and supersonic flow (mach
numbers of 0.50 and 1.77) with no pressure gradient and with heat transfer
to a cold wall .  velocity profiles and average skin-friction
coefficients were calculated .  effects on the velocity profile of
surface roughness and flow length are examined ."""
            }
        ),
        # 6th test case
        ("""1369
.T
steady motion of a sphere., oseen's criticism and solution .
.A
.B
.W
steady motion of a sphere., oseens's criticism and solution .
the formula of stokes for the resistance experienced slowly
moving sphere has been employed in physical researches of fundamental
importance, as a means of estimating the size of minute globules of
water, and thence the number of globules contained in a cloud of
given mass .  consequently the conditions of its validity has been much
discussed both from the experimental and from the theoretical side .""",
         {
             "id": 1369,
             "title": "steady motion of a sphere., oseen's criticism and solution .",
             "author": "",
             "B": "",
             "text": """steady motion of a sphere., oseens's criticism and solution .
the formula of stokes for the resistance experienced slowly
moving sphere has been employed in physical researches of fundamental
importance, as a means of estimating the size of minute globules of
water, and thence the number of globules contained in a cloud of
given mass .  consequently the conditions of its validity has been much
discussed both from the experimental and from the theoretical side ."""
         }),
        # 7th test case
        (
            """471
.T
.A
.B
.W""",
            {
                "id": 471,
                "title": "",
                "author": "",
                "B": "",
                "text": ""
            }
        )
    ]
)
def test_init_CranfieldEntry(text: str, expected: dict[str, int|str]):
    entry = CranfieldEntry(text)

    assert entry.id == expected["id"]
    assert entry.title == expected["title"]
    assert entry.author == expected["author"]
    assert entry.B == expected["B"]
    assert entry.text == expected["text"]


def test_init_CranfieldParser():
    cran = CranfieldParser()

    for entry, id_ in zip(cran, range(1, cran.total + 1)):
        assert entry.id == id_
