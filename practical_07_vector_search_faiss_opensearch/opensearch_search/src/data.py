"""
opensearch_search/src/data.py
================================

WHAT THIS FILE HOLDS
----------------------
50 short product-review-style documents across 5 categories (10 each):
Electronics, Home & Kitchen, Books, Clothing, Sports & Outdoors.

WHY PRODUCT REVIEWS, AND WHY A DIFFERENT DATASET FROM faiss_search/?
-------------------------------------------------------------------------
The practical asks us to compare keyword, semantic, and hybrid search --
that comparison is only interesting on data where those three approaches
can actually DISAGREE. Product reviews are a natural fit: real reviews
routinely describe the same idea in different words (a "rain-resistant
coat" and a "waterproof jacket" mean the same thing but share almost no
vocabulary), which is exactly the situation where keyword search
(matches literal words) and semantic search (matches meaning) diverge --
see search.py's docstring for a worked example using this exact pair.

A separate dataset from faiss_search/ (rather than reusing those 100
sentences) keeps each sub-project self-contained and lets this one be
shaped specifically for demonstrating keyword-vs-semantic contrast,
rather than reusing data optimized for a different purpose.
"""

from typing import List, TypedDict


class Document(TypedDict):
    text: str
    category: str


DOCUMENTS: List[Document] = [

     # --- Sports & Outdoors (10) ---
    {"text": "Lionel Messi, the maestro of Rosario, writes his name into the heavens as the greatest player of all time!", "category": "Sports & Outdoors"},
    {"text": "Cristiano Ronaldo, the ultimate footballing machine, defies gravity once more to claim the throne of the greatest of all time!", "category": "Sports & Outdoors"},
    {"text": "Kylian Mbappé electrifies the stadium, a lightning bolt in human form, leaving defenders chasing ghosts!", "category": "Sports & Outdoors"},
    {"text": "Neymar dances on the edge of a needle, painting the pitch with pure, unadulterated Brazilian magic!", "category": "Sports & Outdoors"},
    {"text": "Erling Haaland, the Nordic terminator, strikes with a ruthless, thumping finality that shakes the very stadium!", "category": "Sports & Outdoors"},
    {"text": "Zinedine Zidane glides through the chaos like a ballet dancer in a thunderstorm, pure footballing poetry!", "category": "Sports & Outdoors"},
    {"text": "Real Madrid, the kings of Europe, summon the ghosts of the Bernabéu for another miraculous, late-night resurrection!", "category": "Sports & Outdoors"},
    {"text": "Barcelona weave their intricate, beautiful tapestry of passing, suffocating their opponents with pure footballing philosophy!", "category": "Sports & Outdoors"},
    {"text": "Kaká, with the stride of an angel and the speed of a bullet, tears the defense apart!", "category": "Sports & Outdoors"},
    {"text": "Real Madrid and Barcelona collision course produces a footballing opera, a symphony of fire and ice in El Clásico!", "category": "Sports & Outdoors"},

    # --- Electronics (10) ---
    {"text": "These wireless earbuds have incredible battery life and clear sound.", "category": "Electronics"},
    {"text": "The laptop's cordless mouse pairs instantly without any lag.", "category": "Electronics"},
    {"text": "This bluetooth speaker fills the room with surprisingly deep bass.", "category": "Electronics"},
    {"text": "The smartwatch tracks my steps and heart rate all day long.", "category": "Electronics"},
    {"text": "This tablet's screen is bright enough to use outdoors in direct sun.", "category": "Electronics"},
    {"text": "The charging cable frayed after just two months of daily use.", "category": "Electronics"},
    {"text": "This noise-cancelling headset blocks out nearly all office chatter.", "category": "Electronics"},
    {"text": "The external hard drive transfers files faster than I expected.", "category": "Electronics"},
    {"text": "This webcam delivers crisp video even in low light conditions.", "category": "Electronics"},
    {"text": "The router's signal reaches every corner of my apartment now.", "category": "Electronics"},

    # --- Home & Kitchen (10) ---
    {"text": "This non-stick pan cleans up in seconds, even after burnt eggs.", "category": "Home & Kitchen"},
    {"text": "The blender crushes ice smoothly without leaving any chunks.", "category": "Home & Kitchen"},
    {"text": "This air fryer makes crispy fries with barely any oil.", "category": "Home & Kitchen"},
    {"text": "The vacuum's suction is strong enough to lift pet hair from carpet.", "category": "Home & Kitchen"},
    {"text": "This coffee maker brews a full pot in under five minutes.", "category": "Home & Kitchen"},
    {"text": "The knife set stayed sharp even after months of regular chopping.", "category": "Home & Kitchen"},
    {"text": "This storage container set stacks neatly and seals tightly.", "category": "Home & Kitchen"},
    {"text": "The stand mixer kneads bread dough without straining the motor.", "category": "Home & Kitchen"},
    {"text": "This water filter noticeably improved the taste of our tap water.", "category": "Home & Kitchen"},
    {"text": "The cutting board is thick enough not to warp in the dishwasher.", "category": "Home & Kitchen"},

    # --- Books (10) ---
    {"text": "This mystery novel kept me guessing until the very last page.", "category": "Books"},
    {"text": "The author's prose in this memoir feels honest and unflinching.", "category": "Books"},
    {"text": "This cookbook's recipes are simple enough for a weeknight dinner.", "category": "Books"},
    {"text": "The pacing in this thriller never lets up from chapter one.", "category": "Books"},
    {"text": "This history book explains complex events in approachable language.", "category": "Books"},
    {"text": "The characters in this fantasy series grow in genuinely believable ways.", "category": "Books"},
    {"text": "This self-help book offers practical steps, not just vague advice.", "category": "Books"},
    {"text": "The short story collection surprised me with its dark humor.", "category": "Books"},
    {"text": "This biography paints a vivid picture of the era it covers.", "category": "Books"},
    {"text": "The sequel matched the emotional depth of the first book.", "category": "Books"},

    # --- Clothing (10) ---
    {"text": "This rain-resistant coat kept me completely dry during the storm.", "category": "Clothing"},
    {"text": "The running shoes feel lightweight but still cushion every stride.", "category": "Clothing"},
    {"text": "This wool sweater is warm without feeling bulky or itchy.", "category": "Clothing"},
    {"text": "The jeans held their shape wash after wash without stretching out.", "category": "Clothing"},
    {"text": "This jacket's zipper jammed within the first week of wear.", "category": "Clothing"},
    {"text": "The dress shirt's fabric stays wrinkle-free through a full workday.", "category": "Clothing"},
    {"text": "This backpack's straps distribute weight evenly on long walks.", "category": "Clothing"},
    {"text": "The socks are thick enough for winter hikes without feeling heavy.", "category": "Clothing"},
    {"text": "This hat's brim blocks sun without obstructing peripheral vision.", "category": "Clothing"},
    {"text": "The gloves are warm enough for cold mornings but still touchscreen-friendly.", "category": "Clothing"},
]


def get_documents() -> List[Document]:
    """Returns the full list of {text, category} documents, in the fixed
    order defined above -- this order is what indexer.py uses to line up
    each document with its corresponding embedding."""
    return DOCUMENTS
