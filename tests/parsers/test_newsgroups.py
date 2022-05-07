from pathlib import Path
from itertools import groupby

import pytest

from src.parsers.newsgroups import NewsgroupsParser, NewsgroupsEntry


@pytest.mark.parametrize(
    "path, expected",
    [
        # 1rst test case
        (
            Path("datasets/20newsgroups-18828/comp.sys.ibm.pc.hardware/60359"),
            {
                "id": 60359,
                "group": "comp.sys.ibm.pc.hardware",
                "from_": "lynn@vax1.mankato.msus.edu",
                "subject": "IDE & MFM in same machine?  HOW?",
                "text": """If anyone out there can help, I would greatly appreciate it.

This christmas, I built a computer out of used parts for my Father-in-law.
The disk drive that I installed was a Seagate 251-1 MFM.  Anyway, he now he
would like to put another HD into this system.  I DON'T want to buy another
MFM, the only reason why I used an MFM in the first place is that it was
FREE.  Would I need a special IDE HD controller?  Also, if I do need a 
special IDE controller, where can I purchase one, & how much are they?

Please send any responses to:
lynn@vax1.mankato.msus.edu


					Thanks in advance,

					Stan Tyree"""
            }
        ),
        # 2nd test case
        (
            Path("datasets/20newsgroups-18828/rec.motorcycles/103121"),
            {
                "id": 103121,
                "group": "rec.motorcycles",
                "from_": "MJMUISE@1302.watstar.uwaterloo.ca (Mike Muise)",
                "subject": "Re: Drinking and Riding",
                "text": """In article <C4wKBp.B9w@eskimo.com>, maven@eskimo.com (Norman Hamer) writes:
>  What is a general rule of thumb for sobriety and cycling? Couple hours 
> after you "feel" sober? What? Or should I just work with "If I drink 
> tonight, I don't ride until tomorrow"?

1 hr/drink for the first 4 drinks.
1.5 hours/drink for the next 6 drinks.
2 hours/drink for the rest.

These are fairly cautious guidelines, and will work even if you happen to 
have a low tolerance or body mass.
I think the cops and "Don't You Dare Drink & Drive" (tm) commercials will 
usually say 1hr/drink in general, but after about 5 drinks and 5 hrs, you 
could very well be over the legal limit. 
Watch yourself.
-Mike
  ________________________________________________
 / Mike Muise / mjmuise@1302.watstar.uwaterloo.ca \ no quotes, no jokes,
 \ Electrical Engineering, University of Waterloo / no disclaimer, no fear."""
            }
        ),
        # 3rd test case
        (
            Path("datasets/20newsgroups-18828/soc.religion.christian/20551"),
            {
                "id": 20551,
                "group": "soc.religion.christian",
                "from_": "atterlep@vela.acs.oakland.edu (Cardinal Ximenez)",
                "subject": "Re: A question that has bee bothering me.",
                "text": """wquinnan@sdcc13.ucsd.edu (Malcusco) writes:

>Especially as we approach a time when Scientists are trying to match God's 
>ability to create life, we should use the utmost caution.

  I question the implications of this statement; namely, that there are certain
physical acts which are limited to God and that attempting to replicate these
acts is blasphemy against God.  God caused a bush to burn without being
consumed--if I do the same thing, am I usurping God's role?  
  Religious people are threatened by science because it has been systematically
removing the physical "proofs" of God's existence.  As time goes on we have to
rely more and more on faith and the spiritual world to relate to God becuase
science is removing our props.  I don't think this is a bad thing.

Alan Terlep				    "Incestuous vituperousness"
Oakland University, Rochester, MI			
atterlep@vela.acs.oakland.edu				   --Melissa Eggertsen
Rushing in where angels fear to tread."""
            }
        ),
        # 4th test case
        (
            Path("datasets/20newsgroups-18828/talk.politics.misc/176905"),
            {
                "id": 176905,
                "group": "talk.politics.misc",
                "from_": "tak@leland.Stanford.EDU (David William Budd)",
                "subject": "Re: Rodney King Trial, Civil Rights Violations, Double Jeopardy",
                "text": """In article <C50puL.CL4@ncratl.AtlantaGA.NCR.COM> mwilson@ncratl.AtlantaGA.NCR.COM (Mark Wilson) writes:
>In <1993Apr2.182942.22445@husc3.harvard.edu> spanagel@husc11.harvard.edu (David Spanagel) writes:
>


>|Furthermore, what are the specific charges against the four LAPD officers? 
>|Which civil rights or laws are they accused of violating? 
>
>I believe it is a general charge, that is no specific right is mentioned.


I don't think that this is accurate. I believe, and could be wrong, that
there IS a specific right allegedly to have been violated, like the
14th or due process or whatever.

>|What about double jeopardy? Has there been any concern that a verdict
>|against Koon, et al. might be overturned upon appeal because they're being tried
>|again for the same actions? (I thought I heard something on the news about 
>|this.)
>
>The SS has previously ruled that since the seperate governments were in
>essence seperate sovereigns, then double jeopardy does not apply.
>
>(If this is true, then could defendents also be tried under city and
>county governments?)
>
>This mornings paper said that the ACLU has decided to reinstate its
>opposition to this kind of thing. They had earlier suspended their
>opposition while they examined the King case. There might be hope
>for the ACLU after all.
>-- 

Double jeopardy does not apply, but not for the reasons you quote. Double
jeopardy states that a person may not be tried twice on the same charge.
However, the police are not on trial for the crime of excessive force
or assault. They are NOW on trial for the DIFFERENT crime of violating
Mr. King's civil rights. 

AS for the city and county or state trying you more than once, 
it most likely will not happen. This is because cities and states
have separate laws governing behaviour. For example, in some states,
it is an offence to carry marijuana, but not a city offence. Also,
I think murder is against federal, but not some state laws. 

===============================================================================
 !           \                                                                 
 !       1-------1                     
 ! \     1_______1           __1__     "And my mind was filled with wonder,
 !  \    1_______1     /   ____1____    when the evening headlines read:
 !       !   \        / /  1__|_|__1    'Richard Cory went home last night,
 !       !    \/       /   ---------     and put a bullet through his head.'"
         ! /    \/      |   |  \   \                                  
                        |  / \____/|"""
            }
        ),
        # 5th test case
        (
            Path("datasets/20newsgroups-18828/misc.forsale/74720"),
            {
                "id": 74720,
                "group": "misc.forsale",
                "from_": "gt1706a@prism.gatech.EDU (Maureen L. Eagle)",
                "subject": "WANTED Brother P-Touch",
                "text": """As it says, I'm interested in buying one of the little
label-makers, and I can't afford a new one.  Anybody
tired of theirs?

E-mail Maureen gt1706a@prism.gatech.edu


-- 
Maureen L. Eagle
Georgia Institute of Technology, Atlanta Georgia, 30332
uucp:	  ...!{decvax,hplabs,ncar,purdue,rutgers}!gatech!prism!gt1706a
Internet: gt1706a@prism.gatech.edu"""
            }
        )
    ]
)
def test_init_NewsgroupsEntry(path, expected):
    entry = NewsgroupsEntry(path)

    assert entry.id == expected["id"]
    assert entry.group == expected["group"]
    assert entry.from_ == expected["from_"]
    assert entry.subject == expected["subject"]
    assert entry.text == expected["text"]


def test_init_NewsgroupsParser():
    newsgroups = NewsgroupsParser()

    groups = {}
    for entry, group in groupby(newsgroups, key=lambda x: x.group):
        try:
            groups[group].append(entry)
        except KeyError:
            groups[group] = [entry]

    assert len(groups) == 20
