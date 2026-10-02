"""Writes prompts/prompts.json: 70 prompts in each of three categories. Raw completion text,
no chat template. Re-running it reproduces the file byte for byte."""
import json
import random
from pathlib import Path

CODE = [
    ("binary_search", "items: list[int], target: int", "Return the index of target in sorted items, or -1."),
    ("merge_sorted", "a: list[int], b: list[int]", "Merge two sorted lists into one sorted list."),
    ("is_palindrome", "text: str", "Return True if text reads the same backwards, ignoring case and spaces."),
    ("fibonacci", "n: int", "Return the n-th Fibonacci number iteratively."),
    ("flatten", "nested: list", "Flatten an arbitrarily nested list into a flat list."),
    ("word_count", "text: str", "Return a dict mapping each word to its number of occurrences."),
    ("gcd", "a: int, b: int", "Return the greatest common divisor using Euclid's algorithm."),
    ("transpose", "matrix: list[list[float]]", "Return the transpose of a rectangular matrix."),
    ("chunk", "items: list, size: int", "Split items into consecutive chunks of the given size."),
    ("parse_csv_line", "line: str", "Split a CSV line on commas, honouring double-quoted fields."),
    ("rle_encode", "text: str", "Run-length encode a string, e.g. 'aaab' -> 'a3b1'."),
    ("rle_decode", "encoded: str", "Invert rle_encode."),
    ("is_prime", "n: int", "Return True if n is prime, using trial division up to sqrt(n)."),
    ("primes_below", "n: int", "Return all primes below n with the sieve of Eratosthenes."),
    ("dedupe", "items: list", "Remove duplicates while preserving first-seen order."),
    ("rotate", "items: list, k: int", "Rotate items to the right by k positions."),
    ("anagrams", "words: list[str]", "Group words that are anagrams of each other."),
    ("moving_average", "values: list[float], window: int", "Return the moving average over a sliding window."),
    ("to_snake_case", "name: str", "Convert CamelCase to snake_case."),
    ("to_camel_case", "name: str", "Convert snake_case to CamelCase."),
    ("matrix_multiply", "a: list[list[float]], b: list[list[float]]", "Multiply two matrices."),
    ("levenshtein", "a: str, b: str", "Return the edit distance between two strings."),
    ("balanced_brackets", "text: str", "Return True if (), [] and {} are balanced in text."),
    ("roman_to_int", "numeral: str", "Convert a Roman numeral to an integer."),
    ("int_to_roman", "value: int", "Convert an integer from 1 to 3999 to a Roman numeral."),
    ("median", "values: list[float]", "Return the median of a non-empty list."),
    ("mode", "values: list[int]", "Return the most common value; ties go to the smallest."),
    ("histogram", "values: list[int]", "Return a text histogram with one row of '#' per value."),
    ("binary_to_int", "bits: str", "Convert a string of 0s and 1s to an integer."),
    ("int_to_binary", "value: int", "Return the binary representation of a non-negative integer."),
    ("caesar", "text: str, shift: int", "Apply a Caesar cipher to letters, leaving other characters."),
    ("vowel_count", "text: str", "Count the vowels in text."),
    ("longest_common_prefix", "words: list[str]", "Return the longest common prefix of all words."),
    ("two_sum", "nums: list[int], target: int", "Return indices of two numbers that add up to target."),
    ("max_subarray", "nums: list[int]", "Return the largest sum of a contiguous subarray (Kadane)."),
    ("bfs", "graph: dict[int, list[int]], start: int", "Return nodes in breadth-first order."),
    ("dfs", "graph: dict[int, list[int]], start: int", "Return nodes in depth-first order."),
    ("topological_sort", "graph: dict[str, list[str]]", "Return a topological order or raise on a cycle."),
    ("dijkstra", "graph: dict[str, dict[str, float]], source: str", "Return shortest distances from source."),
    ("lru_get", "cache: dict, order: list, key: str", "Return a cached value and mark it most recently used."),
    ("parse_query_string", "query: str", "Parse 'a=1&b=2' into a dict of strings."),
    ("format_bytes", "size: int", "Format a byte count as a human-readable string like '1.5 MB'."),
    ("clamp", "value: float, low: float, high: float", "Clamp value into [low, high]."),
    ("lerp", "a: float, b: float, t: float", "Linearly interpolate between a and b."),
    ("dot", "a: list[float], b: list[float]", "Return the dot product of two vectors."),
    ("normalize", "v: list[float]", "Return v scaled to unit length."),
    ("factorial", "n: int", "Return n! for a non-negative integer."),
    ("power_set", "items: list", "Return all subsets of items."),
    ("permutations", "items: list", "Return all permutations of items."),
    ("pascal_row", "n: int", "Return the n-th row of Pascal's triangle."),
    ("count_islands", "grid: list[list[int]]", "Count connected groups of 1s in a grid."),
    ("valid_ipv4", "address: str", "Return True if address is a valid dotted IPv4 address."),
    ("slugify", "title: str", "Lowercase, strip punctuation and join words with hyphens."),
    ("wrap_text", "text: str, width: int", "Wrap text into lines no longer than width."),
    ("tail", "path: str, n: int", "Return the last n lines of a text file."),
    ("merge_intervals", "intervals: list[tuple[int, int]]", "Merge overlapping intervals."),
    ("quicksort", "items: list[int]", "Sort items with quicksort and return a new list."),
    ("mergesort", "items: list[int]", "Sort items with merge sort and return a new list."),
    ("insertion_sort", "items: list[int]", "Sort items in place with insertion sort."),
    ("bisect_left", "items: list[int], x: int", "Return the leftmost insertion point for x."),
    ("running_total", "values: list[int]", "Return the cumulative sums of values."),
    ("zip_longest", "a: list, b: list, fill: object", "Pair items, padding the shorter list with fill."),
    ("char_frequency", "text: str", "Return characters sorted by descending frequency."),
    ("is_leap_year", "year: int", "Return True if year is a Gregorian leap year."),
    ("days_between", "a: str, b: str", "Return days between two ISO dates."),
    ("celsius_to_fahrenheit", "c: float", "Convert Celsius to Fahrenheit."),
    ("parse_duration", "text: str", "Parse strings like '1h30m' into seconds."),
    ("safe_divide", "a: float, b: float", "Divide, returning None when b is zero."),
    ("unique_paths", "rows: int, cols: int", "Count lattice paths from top-left to bottom-right."),
    ("coin_change", "coins: list[int], amount: int", "Return the fewest coins that make amount, or -1."),
]

PROSE = [
    "The lighthouse keeper had not spoken to anyone in eleven days when the boat appeared on the horizon.",
    "Most cities were not planned; they accreted, one road and one argument at a time.",
    "Bread is one of the oldest technologies humans still use every day.",
    "When the river froze that winter, the whole village walked across it to the market.",
    "Sleep researchers have long puzzled over why animals spend so much of their lives unconscious.",
    "My grandmother kept every letter she ever received in a tin box under her bed.",
    "The first thing you notice about the desert at night is the silence.",
    "Glaciers move, but so slowly that a lifetime is not long enough to see it.",
    "The museum's oldest exhibit is a piece of pottery that nobody can date with confidence.",
    "On the morning of the election, the queues began forming before sunrise.",
    "Octopuses solve puzzles in ways that still surprise the people who study them.",
    "The train was late again, and the platform had become a small, irritable community.",
    "Paper maps teach a kind of attention that turn-by-turn directions quietly remove.",
    "In the spring of that year, the orchard bloomed three weeks early.",
    "Some languages have no word for left or right, only for the points of the compass.",
    "The old bridge had survived two floods and one very determined demolition crew.",
    "Every library smells slightly different, and regular readers can tell them apart.",
    "The chess match lasted six hours and ended in a draw that satisfied nobody.",
    "Coffee reached Europe as a medicine before it became a habit.",
    "She learned to sail at forty, which she later said was the right age to be afraid of the sea.",
    "The town clock had been wrong for so long that people set their watches by its error.",
    "Volcanic soil is fertile, which is why people keep returning to live beside volcanoes.",
    "The recipe had been handed down for four generations, and each one had changed it.",
    "A good teacher remembers what it was like not to understand.",
    "The storm took the roof off the school but left the piano untouched.",
    "Bees communicate the location of flowers through a dance whose meaning took decades to decode.",
    "He kept a notebook of overheard sentences and never explained why.",
    "The hospital at night runs on a different rhythm from the hospital by day.",
    "Salt was once valuable enough that soldiers were said to be paid in it.",
    "The expedition carried enough food for ninety days and needed it for a hundred and twelve.",
    "Mountains create their own weather, and climbers learn to read it or turn back.",
    "Her first novel was rejected thirty times before a small press accepted it.",
    "Street names record a city's history more faithfully than its monuments do.",
    "The fishing fleet left before dawn, as it had every day for three hundred years.",
    "Children invent rules for their games faster than adults can write them down.",
    "The telescope revealed that the faint smudge was not one star but thousands.",
    "Moving house is partly an inventory of everything you forgot you owned.",
    "The violin maker spent a month choosing the wood before making a single cut.",
    "Tides are the moon's most visible effect on daily life.",
    "The negotiation stalled over a single sentence in the third paragraph.",
    "Wolves were reintroduced to the park, and within a decade the rivers changed course.",
    "The bakery opened at five, and by six the line reached the corner.",
    "Translators often say the hardest words are the simplest ones.",
    "The power went out across the valley just as the final match began.",
    "Old photographs rarely show people smiling, and the reason is partly technical.",
    "The garden had been neglected for years, but the roses had not noticed.",
    "A single misplaced decimal point cost the company a great deal of money.",
    "Migratory birds navigate using the stars, the sun, and the earth's magnetic field.",
    "The village had one telephone, and it lived in the post office.",
    "Learning a second language changes how you hear your first.",
    "The ship's log recorded the weather every four hours for the entire voyage.",
    "Forests recover from fire in stages that ecologists can predict surprisingly well.",
    "The interview went well until the final question.",
    "Every winter the same family of foxes returned to the abandoned barn.",
    "Handwriting is disappearing from schools, and not everyone thinks that is a loss.",
    "The marathon route passed through four neighbourhoods that rarely met.",
    "Ancient roads were built to last, and some are still in use today.",
    "The orchestra tuned to a single note from the oboe.",
    "A drought reveals the outlines of villages flooded long ago by reservoirs.",
    "The detective noticed that the clock on the mantelpiece had stopped at a quarter past nine.",
    "Clouds look solid from above, which is one of the first surprises of flying.",
    "The cooperative began with six farmers and a borrowed truck.",
    "Earthquakes are measured on a scale that most people misunderstand.",
    "The children's hospital painted every ceiling, because that is what patients look at.",
    "Her research began with a footnote that did not add up.",
    "In the archive, the most interesting documents were the ones nobody had catalogued.",
    "The ferry crossing took forty minutes, long enough for strangers to start talking.",
    "Ice cores hold a record of the atmosphere going back hundreds of thousands of years.",
    "The festival had been cancelled twice, and this year the town was determined.",
    "Nobody remembered who first planted the oak in the middle of the square.",
]


def templated(rng: random.Random) -> list[str]:
    names = ["Alice", "Bob", "Carmen", "Deniz", "Elif", "Farid", "Grace", "Hiro", "Ines", "Jonas", "Kemal", "Lena"]
    cities = ["Ankara", "Lisbon", "Oslo", "Lima", "Osaka", "Nairobi", "Quito", "Riga", "Tunis", "Perth"]
    items = ["apples", "bolts", "cables", "drills", "envelopes", "filters", "gloves", "hinges", "inks", "jars"]
    out: list[str] = []
    for i in range(7):
        rows = rng.sample(names, 4)
        out.append("id,name,city,age\n" + "".join(f"{j + 1},{n},{rng.choice(cities)},{rng.randint(20, 70)}\n" for j, n in enumerate(rows[:3])) + f"4,{rows[3]},")
    for i in range(7):
        recs = [f'  {{"sku": "{rng.choice(items)[:3].upper()}-{rng.randint(100, 999)}", "qty": {rng.randint(1, 50)}, "unit": "box"}},\n' for _ in range(3)]
        out.append("[\n" + "".join(recs) + '  {"sku": "')
    for i in range(7):
        start = rng.randint(1, 20)
        out.append("".join(f"Day {d}: Wake up, run {rng.randint(3, 10)} km, write {rng.randint(200, 900)} words.\n" for d in range(start, start + 3)) + f"Day {start + 3}: Wake up, run")
    for i in range(7):
        n = rng.choice(names)
        out.append(f"Dear {n},\n\nThank you for your order #{rng.randint(1000, 9999)}. Your {rng.choice(items)} will ship on Monday.\n\nBest regards,\nThe Shipping Team\n\nDear {rng.choice(names)},\n\nThank you for your order #")
    for i in range(7):
        t = rng.randint(0, 50)
        out.append("".join(f"2026-10-02 09:{t + j:02d}:00 INFO worker-{j % 3} processed batch {100 + j} in {rng.randint(10, 99)} ms\n" for j in range(4)) + f"2026-10-02 09:{t + 4:02d}:00 INFO worker-")
    for base in rng.sample(range(2, 20), 7):
        out.append("".join(f"{base} x {j} = {base * j}\n" for j in range(1, 5)) + f"{base} x 5 =")
    for i in range(7):
        table = rng.choice(["users", "orders", "products"])
        out.append("".join(f"INSERT INTO {table} (id, name, score) VALUES ({j}, '{rng.choice(names)}', {rng.randint(1, 100)});\n" for j in range(1, 4)) + f"INSERT INTO {table} (id, name, score) VALUES (4, '")
    for i in range(7):
        out.append("<ul>\n" + "".join(f"  <li class=\"item\"><a href=\"/{c.lower()}\">{c}</a></li>\n" for c in rng.sample(cities, 3)) + "  <li class=\"item\"><a href=\"/")
    for i in range(7):
        q = rng.sample(names, 4)
        out.append("".join(f"Q: What is {n}'s favourite colour?\nA: {n}'s favourite colour is {rng.choice(['red', 'blue', 'green', 'amber'])}.\n\n" for n in q[:3]) + f"Q: What is {q[3]}'s favourite colour?\nA:")
    for i in range(7):
        out.append("".join(f"- [ ] Review pull request #{rng.randint(10, 99)} from {rng.choice(names)}\n" for _ in range(4)) + "- [ ] Review pull request #")
    return out


def main() -> None:
    rng = random.Random(0)
    code = [f'def {name}({args}):\n    """{doc}"""\n' for name, args, doc in CODE]
    prompts = {"code": code, "prose": PROSE, "repetitive": templated(rng)}
    for category, items in prompts.items():
        assert len(items) == 70 and len(set(items)) == 70, category
    Path(__file__).with_name("prompts.json").write_text(json.dumps(prompts, indent=1) + "\n")


if __name__ == "__main__":
    main()
