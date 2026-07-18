"""
faiss_search/src/data.py
===========================

WHAT THIS FILE HOLDS
----------------------
100 example sentences across 5 topics (20 each): Sports, Technology,
Food & Cooking, Travel & Places, Health & Wellness.

WHY 100 SENTENCES, AND WHY 5 TOPICS INSTEAD OF THE 4 USED ELSEWHERE?
------------------------------------------------------------------------
The practical brief specifically asks for 100 embeddings, larger than a
"hello world"-sized set -- large enough that FAISS's exact search still
runs instantly, but big enough to make "find the best match" a genuinely
non-trivial question (with only a handful of sentences, almost anything
looks "close enough"). A 5th topic (Health & Wellness) was added
alongside the more common 4 (Sports/Tech/Food/Travel) purely to reach
100 at an even 20-per-topic split, so results stay easy to sanity-check
by category, rather than an arbitrary uneven count.
"""

from typing import List, TypedDict


class LabeledSentence(TypedDict):
    sentence: str
    topic: str


SENTENCE_DATA: List[LabeledSentence] = [
    # --- Football(20) ---
    {"sentence": "Lionel Messi, the diminutive magician of Rosario, weaves through a forest of legs to paint his masterpiece upon the canvas of football history!", "topic": "Football"},
    {"sentence": "Cristiano Ronaldo, a towering monument of athletic perfection, leaps into the heavens to plant a thunderous header into the net!", "topic": "Football"},
    {"sentence": "Kylian Mbappé ignites his afterburners, a streak of pure, unadulterated lightning leaving gasping defenders in his wake!", "topic": "Football"},
    {"sentence": "Neymar dances upon the edge of a penny, treating the ball like a loyal friend in a Samba of pure defiance!", "topic": "Football"},
    {"sentence": "Erling Haaland, the Nordic colossus, strikes with a violent, predatory instinct that tears the very fabric of the net!", "topic": "Football"},
    {"sentence": "Zinedine Zidane glides through the midfield storm like a majestic galleon, turning brutal chaos into elegant, timeless poetry!", "topic": "Football"},
    {"sentence": "Real Madrid, the eternal kings of Europe, invoke the mystic spirits of the Bernabéu for yet another breathless, midnight resurrection!", "topic": "Football"},
    {"sentence": "Barcelona weave their hypnotic tapestry of passing, a beautiful, suffocating symphony of tiki-taka that leaves opponents chasing ghosts!", "topic": "Football"},
    {"sentence": "Kaká, with the stride of an angel and the velocity of a bullet, glides past the despairing challenges of mortal men!", "topic": "Football"},
    {"sentence": "The whistle blows and El Clásico erupts, a footballing opera of fire and ice, splitting the sporting cosmos in two!", "topic": "Football"},
    {"sentence": "The goalkeeper flies through the air like a desperate swallow, tipping the ball over the bar with a fingertip salvation!", "topic": "Football"},
    {"sentence": "Silence grips the arena as the maestro steps up to the penalty spot, carrying the weight of a million beating hearts!", "topic": "Football"},
    {"sentence": "The referee points to the center circle, and the stadium dissolves into a wild, intoxicating sea of absolute, unbridled ecstasy!", "topic": "Football"},
    {"sentence": "They have climbed the mountain, they have conquered the continent, and the sky bleeds the triumphant colors of the champions!", "topic": "Football"},
    {"sentence": "A tackle of pure steel and immaculate timing stops the rampaging striker dead in his tracks, a heroic act of defensive defiance!", "topic": "Football"},
    {"sentence": "He strikes it from distance, a screaming missile of hope that dips, swerves, and crashes into the top corner of the net!", "topic": "Football"},
    {"sentence": "The manager paces the technical area like a caged tiger, kicking every ball and feeling every agonizing tick of the clock!", "topic": "Football"},
    {"sentence": "It is a tactical chess match played at supersonic speed, two footballing masterminds locked in a deadly, intellectual embrace!", "topic": "Football"},
    {"sentence": "The stadium shakes to its very foundations as the home supporters unleash a wall of sound to carry their heroes home!", "topic": "Football"},
    {"sentence": "In the dying, breathless seconds of stoppage time, destiny finds its man, and the beautiful game delivers its ultimate, cruel twist!", "topic": "Football"},


    # --- Technology (20) ---
    {"sentence": "The new smartphone features a faster processor and better camera.", "topic": "Technology"},
    {"sentence": "Developers are increasingly using AI to write and review code.", "topic": "Technology"},
    {"sentence": "The startup raised funding to build its cloud infrastructure.", "topic": "Technology"},
    {"sentence": "A software update fixed the security vulnerability overnight.", "topic": "Technology"},
    {"sentence": "The company unveiled a robot capable of folding laundry.", "topic": "Technology"},
    {"sentence": "Engineers are testing a chip that runs cooler at higher speeds.", "topic": "Technology"},
    {"sentence": "The app's redesign focused on faster load times.", "topic": "Technology"},
    {"sentence": "A data breach exposed millions of user records last week.", "topic": "Technology"},
    {"sentence": "The team migrated the database to a new cloud provider.", "topic": "Technology"},
    {"sentence": "Researchers trained a model to detect defects on assembly lines.", "topic": "Technology"},
    {"sentence": "The browser extension blocks trackers automatically.", "topic": "Technology"},
    {"sentence": "Engineers deployed the new app update to all users overnight.", "topic": "Technology"},
    {"sentence": "The wearable device tracks sleep patterns and heart rate.", "topic": "Technology"},
    {"sentence": "A new programming language promises simpler concurrency.", "topic": "Technology"},
    {"sentence": "The satellite relayed data back to the ground station in seconds.", "topic": "Technology"},
    {"sentence": "The autonomous vehicle navigated the intersection without incident.", "topic": "Technology"},
    {"sentence": "The keyboard shortcut saved the developer several minutes a day.", "topic": "Technology"},
    {"sentence": "The server rack overheated during the stress test.", "topic": "Technology"},
    {"sentence": "The open-source project gained thousands of contributors this year.", "topic": "Technology"},
    {"sentence": "The chatbot answered most support tickets without human help.", "topic": "Technology"},

    {"sentence": "Virat Kohli steps down the track and lofts it high into the night sky, that is absolutely massive, what a shot!", "topic": "Cricket"},
    {"sentence": "The Indian Cricket Team has done it, they have conquered the world and the celebrations are going to go all night long!", "topic": "Cricket"},
    {"sentence": "Rohit Sharma pulls it away with supreme authority, it is into the crowd, he is hitting them like a tracer bullet today!", "topic": "Cricket"},
    {"sentence": "Dhoni finishes it off in style, a magnificent strike into the crowd, India lift the World Cup after twenty-eight years!", "topic": "Cricket"},
    {"sentence": "Jasprit Bumrah bowls a devastating, toe-crushing yorker, the stumps are a total mess, he is just absolute gold for India!", "topic": "Cricket"},
    {"sentence": "Hardik Pandya stands tall and delivers a monstrous hit over long-on, the crowd is going absolutely berserk here!", "topic": "Cricket"},
    {"sentence": "He edges it and gone, taken in the slips, India strike early and the pressure is right back on the opposition!", "topic": "Cricket"},
    {"sentence": "The umpire raises the finger and that is a massive, massive wicket, the stadium has gone completely silent!", "topic": "Cricket"},
    {"sentence": "A sensational diving catch at the boundary ropes, he has pulled that out of thin air, absolutely unbelievable athleticism!", "topic": "Cricket"},
    {"sentence": "He brings up his century with a glorious cover drive, a masterclass in batting under immense, high-octane pressure!", "topic": "Cricket"},
    {"sentence": "Three runs needed off the final ball, the bowler runs in, what a dramatic finish we have on our hands today!", "topic": "Cricket"},
    {"sentence": "He cleans him up, through the gate, the crowd is up on their feet as the bowling side celebrates a famous victory!", "topic": "Cricket"},
    {"sentence": "A direct hit at the bowler's end and he is well short of his crease, a phenomenal piece of fielding changes the game!", "topic": "Cricket"},
    {"sentence": "The rain is coming down hard now, the groundstaff are rushing onto the field with the covers, what a nervous wait!", "topic": "Cricket"},
    {"sentence": "He wins the toss and chooses to bat first on a pitch that looks like an absolute paradise for the batsmen!", "topic": "Cricket"},
    {"sentence": "Mumbai Indians pull off a dramatic final-over jailbreak at the Wankhede, the sea of blue is absolutely rocking!", "topic": "Cricket"},
    {"sentence": "RCB lock and load for another high-voltage chase at the Chinnaswamy, the noise here is truly deafening!", "topic": "Cricket"},
    {"sentence": "Young prodigy Vaibhav Sooryavanshi shows nerves of steel, smashing the ball past the fielder with incredible maturity!", "topic": "Cricket"},
    {"sentence": "CSK turn the game right on its head with a tactical masterstroke, MS Dhoni working his magic behind the stumps!", "topic": "Cricket"},
    {"sentence": "It is the El Clásico of the IPL, MI versus CSK, a clash of titans where reputation counts for absolutely nothing!", "topic": "Cricket"},


    # --- Travel & Places (20) ---
    {"sentence": "The train wound through the mountains toward the coastal town.", "topic": "Travel & Places"},
    {"sentence": "Tourists gathered at sunset to watch the view from the old fort.", "topic": "Travel & Places"},
    {"sentence": "She backpacked across three countries in a single summer.", "topic": "Travel & Places"},
    {"sentence": "The airline announced a new direct route to the island.", "topic": "Travel & Places"},
    {"sentence": "The old quarter's narrow streets are best explored on foot.", "topic": "Travel & Places"},
    {"sentence": "The ferry crossed the strait just as the fog began to lift.", "topic": "Travel & Places"},
    {"sentence": "Hikers reached the summit just before the storm rolled in.", "topic": "Travel & Places"},
    {"sentence": "The hotel overlooked a quiet bay dotted with fishing boats.", "topic": "Travel & Places"},
    {"sentence": "The road trip took them through three national parks.", "topic": "Travel & Places"},
    {"sentence": "The market stalls overflowed with spices and woven textiles.", "topic": "Travel & Places"},
    {"sentence": "The city's skyline glittered from across the river at night.", "topic": "Travel & Places"},
    {"sentence": "They pitched their tent beside a lake ringed by pine trees.", "topic": "Travel & Places"},
    {"sentence": "The guide led the group through the ruins at dawn.", "topic": "Travel & Places"},
    {"sentence": "The night train rattled past fields lit only by the moon.", "topic": "Travel & Places"},
    {"sentence": "The village square filled with music during the festival.", "topic": "Travel & Places"},
    {"sentence": "The desert stretched out in waves of red and gold dunes.", "topic": "Travel & Places"},
    {"sentence": "A cable car carried visitors up to the mountaintop monastery.", "topic": "Travel & Places"},
    {"sentence": "The coastal path wound past cliffs and hidden coves.", "topic": "Travel & Places"},
    {"sentence": "The airport terminal buzzed with travelers rushing to connections.", "topic": "Travel & Places"},
    {"sentence": "They wandered the canals of the old town until well past midnight.", "topic": "Travel & Places"},

    # --- Health & Wellness (20) ---
    {"sentence": "She meditates for ten minutes every morning before work.", "topic": "Health & Wellness"},
    {"sentence": "Doctors recommend at least seven hours of sleep a night.", "topic": "Health & Wellness"},
    {"sentence": "He switched to a plant-based diet after his checkup.", "topic": "Health & Wellness"},
    {"sentence": "The physical therapist designed a routine to strengthen his knee.", "topic": "Health & Wellness"},
    {"sentence": "Regular stretching has noticeably improved her flexibility.", "topic": "Health & Wellness"},
    {"sentence": "The clinic offers free screenings during health awareness week.", "topic": "Health & Wellness"},
    {"sentence": "He tracks his water intake to stay hydrated throughout the day.", "topic": "Health & Wellness"},
    {"sentence": "Deep breathing exercises help calm her before a big presentation.", "topic": "Health & Wellness"},
    {"sentence": "The gym added a new class focused on mobility and balance.", "topic": "Health & Wellness"},
    {"sentence": "Cutting back on sugar improved his energy levels within weeks.", "topic": "Health & Wellness"},
    {"sentence": "The nutritionist suggested smaller, more frequent meals.", "topic": "Health & Wellness"},
    {"sentence": "Walking after dinner has become part of their evening routine.", "topic": "Health & Wellness"},
    {"sentence": "The therapist recommended journaling to manage daily stress.", "topic": "Health & Wellness"},
    {"sentence": "Her doctor advised more magnesium-rich foods for better sleep.", "topic": "Health & Wellness"},
    {"sentence": "The wellness app reminds users to stand up every hour.", "topic": "Health & Wellness"},
    {"sentence": "He started strength training to support his joints as he ages.", "topic": "Health & Wellness"},
    {"sentence": "Mindfulness practice has helped reduce her anxiety at work.", "topic": "Health & Wellness"},
    {"sentence": "The community center runs a free yoga session on weekends.", "topic": "Health & Wellness"},
    {"sentence": "Staying consistent with a bedtime routine improved his sleep quality.", "topic": "Health & Wellness"},
    {"sentence": "She swapped her afternoon coffee for a short walk outside.", "topic": "Health & Wellness"},
]


def get_sentences() -> List[str]:
    """Returns just the sentence text, in the fixed order defined above."""
    return [item["sentence"] for item in SENTENCE_DATA]


def get_topics() -> List[str]:
    """Returns just the topic labels, in the SAME order as
    get_sentences() -- useful for grouping/inspecting search results by
    category, e.g. checking whether a query's top-5 matches all came from
    the topic you'd expect."""
    return [item["topic"] for item in SENTENCE_DATA]
