import hashlib
import logging
from pathlib import Path
import concurrent.futures
from typing import Optional
import math
from functools import partial

logger = logging.getLogger(__name__)


class ProjectVersion:
    words_list = [
        "Alfa", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel",
        "Juliette", "Kilo", "Lima", "Mike", "November", "Oscar", "Papa",
        "Romeo", "Sierra", "Tango", "Uniform", "Victor", "Whiskey",
        "X-ray", "Yankee", "Zulu", "Apple", "Banana", "Cherry", "Date", "Elder",
        "Fig", "Grape", "Honey", "Kiwi", "Lemon", "Mango", "Nectarine", "Olive",
        "Peach", "Quince", "Raspberry", "Strawberry", "Tangerine", "Valencia",
        "Watermelon", "Xigua", "Yam", "Zucchini", "Ant", "Bear", "Cat", "Dog",
        "Eagle", "Fox", "Giraffe", "Hippo", "Iguana", "Jaguar", "Kangaroo", "Lion",
        "Monkey", "Newt", "Owl", "Penguin", "Quail", "Rabbit", "Shark", "Turtle",
        "Urchin", "Vulture", "Walrus", "Yak", "Zebra", "Apricot", "Blueberry",
        "Cantaloupe", "Dragonfruit", "Elderberry", "Guava", "Honeydew", "Jackfruit",
        "Kumquat", "Lime", "Mulberry", "Nutmeg", "Passion", "Radish", "Squash",
        "Orange", "Vanilla", "Watercress", "Yarrow", "Amaranth", "Basil",
        "Chive", "Dill", "Endive", "Fennel", "Garlic", "Horseradish", "Iceberg",
        "Jicama", "Kale", "Lettuce", "Mint", "Nori", "Oregano", "Parsley", "Quinoa",
        "Radicchio", "Sage", "Thyme", "Upland", "Vetch", "Wasabi", "Almond",
        "Cashew", "Chestnut", "Hazelnut", "Macadamia", "Peanut", "Pecan", "Pistachio",
        "Walnut", "Amethyst", "Aquamarine", "Diamond", "Emerald", "Fire Opal",
        "Garnet", "Onyx", "Pearl", "Quartz", "Ruby", "Sapphire", "Topaz",
        "Turquoise", "Zircon", "Abacus", "Bell", "Candle", "Drum", "Egg", "Flute",
        "Guitar", "Harmonica", "Icicle", "Jar", "Kite", "Lantern", "Mirror",
        "Notebook", "Oboe", "Paintbrush", "Quilt", "Ruler", "Scissors", "Tambourine",
        "Ukulele", "Vase", "Whistle", "Xylophone", "Yarn", "Zipper", "Acorn",
        "Bookmark", "Compass", "Dice", "Envelope", "Feather", "Globe", "Hammer",
        "Ink", "Journal", "Key", "Lock", "Magnet", "Nail", "Ornament", "Paperclip",
        "Quill", "Stapler", "Ticket", "Umbrella", "Violin", "Wheel",
        "Yo-yo", "Asphalt", "Brick", "Cement", "Dirt", "Earth", "Foam", "Granite",
        "Herb", "Iron", "Jade", "Kelp", "Limestone", "Marble", "Nectar", "Opal",
        "Plaster", "Rock", "Sand", "Tile", "Underlay", "Vine", "Wool",
        "Yardstick", "Zenith", "Arch", "Beam", "Column", "Dome", "Edge", "Floor",
        "Gable", "Height", "Isle", "Joy", "Keystone", "Ledge", "Mantle", "Nook",
        "Oriel", "Parapet", "Quoin", "Rafter", "Stair", "Truss", "Upper", "Vault",
        "Wall", "Zone", "Act", "Ban", "Cast", "Drift", "Robot"
    ]

    adjectives = [
        "Bright", "Cheerful", "Lively", "Sunny", "Vivid", "Zesty", "Radiant", "Sparkling",
        "Brisk", "Crisp", "Daring", "Fearless", "Valiant", "Nimble", "Quick", "Snappy",
        "Clever", "Witty", "Sassy", "Quirky", "Playful", "Jolly", "Bubbly",
        "Smooth", "Sleek", "Glossy", "Polished", "Velvety", "Silky", "Lustrous", "Gleaming",
        "Gentle", "Tender", "Soft", "Mellow", "Calm", "Soothing", "Serene", "Peaceful",
        "Bold", "Dashing", "Gallant", "Stalwart", "Steady", "Dependable", "Reliable", "Loyal",
        "Sharp", "Astute", "Keen", "Perceptive", "Insightful", "Intuitive", "Shrewd",
        "Friendly", "Warm", "Kind", "Affectionate", "Caring", "Compassionate", "Helpful", "Supportive",
        "Elegant", "Graceful", "Charming", "Sophisticated", "Refined", "Dignified", "Poised",
        "Energetic", "Vigorous", "Dynamic", "Animated", "Excited", "Peppy", "Bouncy",
        "Unique", "Original", "Inventive", "Creative", "Imaginative", "Artistic", "Visionary", "Innovative",
        "Hardy", "Tough", "Rugged", "Robust", "Sturdy", "Resilient", "Enduring", "Unyielding",
        "Cheeky", "Jovial", "Whimsical", "Zany", "Wacky",
        "Majestic", "Grand", "Noble", "Regal", "Stately", "Magnificent", "Opulent", "Splendid",
        "Fresh", "Clean", "Pure", "Pristine", "Unspoiled", "Untouched", "Spotless",
        "Fast", "Rapid", "Swift", "Speedy", "Zippy", "Agile",
        "Luminous", "Glowing", "Shiny", "Brilliant", "Blazing", "Beaming", "Dazzling",
        "Silly", "Goofy", "Quaint", "Merry",
        "Snug", "Cozy", "Comfy", "Inviting", "Welcoming", "Cuddly",
        "Wise", "Sage", "Judicious", "Prudent", "Thoughtful",
        "Curious", "Inquisitive", "Eager", "Restless", "Questioning", "Investigative", "Observant", "Analytical"
    ]

    def __init__(self, directory_path: Path, algorithm: str = "md5"):
        """
        Initialise the ProjectVersion with the path to the directory.

        Args:
            directory_path (Path): Path to the directory containing Python files.
            algorithm (str): The hashing algorithm to use ('md5' or 'sha256').
        """
        self.directory_path = Path(directory_path)
        if not self.directory_path.exists() or not self.directory_path.is_dir():
            raise ValueError(f"Provided path '{self.directory_path}' is not a valid directory.")

        self.algorithm = algorithm.lower()
        if self.algorithm not in {"md5", "sha256"}:
            raise ValueError("Unsupported algorithm. Please use 'md5' or 'sha256'.")

    def _hash_file_parallel(self, file_path: Path) -> Optional[str]:
        """
        Compute the hash of a single file using the specified algorithm.

        Args:
            file_path (Path): The path to the file.

        Returns:
            Optional[str]: The hex digest of the file's hash, or None if an error occurs.
        """
        try:
            hasher = (
                hashlib.sha256(usedforsecurity=False)
                if self.algorithm == "sha256"
                else hashlib.md5(usedforsecurity=False)
            )
            # Using functools.partial to avoid recreating the lambda every iteration.
            with file_path.open("rb") as f:
                for chunk in iter(partial(f.read, 8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.exception(f"Error processing file {file_path}: {e}")
            return None

    def hash(self, accepted_collision_probability: float = 0.001) -> str:
        """
        Calculate a unique version hash for all Python files in the directory and its subdirectories based on their contents.

        Args:
            accepted_collision_probability (float): The accepted probability of a hash collision (0 < p < 1).
        Returns:
            str: A unique hash string or its phonetic representation.
        """

        # Gather sorted list of Python files.
        files = sorted(self.directory_path.rglob("*.py"))

        file_count = len(files)
        if file_count == 0:
            raise ValueError("No Python files found in the directory.")

        logger.debug(f"Found {file_count} Python files in the directory.")

        # calculate the number of word pairs needed to support the given file count
        length = -1
        if accepted_collision_probability > 0:
            length = self.min_word_pairs_needed(file_count, accepted_collision_probability)
            logger.debug(f"Minimum word pairs needed to support {file_count} files with "
                         f"collision probability of {accepted_collision_probability}: {length}")

        # Process files in parallel. Using executor.map preserves order.
        with concurrent.futures.ThreadPoolExecutor() as executor:
            file_hashes = list(executor.map(self._hash_file_parallel, files))

        # Combine individual file hashes in a deterministic order.
        final_hasher = (
            hashlib.sha256(usedforsecurity=False)
            if self.algorithm == "sha256"
            else hashlib.md5(usedforsecurity=False)
        )

        # Zip the sorted file paths with their corresponding hashes.
        for file_path, file_hash in zip(files, file_hashes):
            if file_hash is not None:
                final_hasher.update(str(file_path).encode("utf-8"))
                final_hasher.update(file_hash.encode("utf-8"))

        version_hash = final_hasher.hexdigest()

        # Calculate total possible unique pairs.
        total_combinations = len(self.adjectives) * len(self.words_list)
        # Determine how many bits we can uniquely encode.
        max_bits = math.floor(math.log2(total_combinations))
        # Calculate the number of hex digits needed (each hex digit represents 4 bits).
        chunk_size = math.ceil(max_bits / 4)
        words = []
        for i in range(0, len(version_hash), chunk_size):
            hex_group = version_hash[i: i + chunk_size].ljust(chunk_size, "0")
            value = int(hex_group, 16) % total_combinations
            adj_index = value // len(self.words_list)
            word_index = value % len(self.words_list)
            words.append(f"{self.adjectives[adj_index]}-{self.words_list[word_index]}")

        if length > 0:
            words = words[:length]
        return " ".join(words)

    def min_word_pairs_needed(self, file_count: int, accepted_collision_prob: float) -> int:
        """
        Calculate the minimum number of word pairs required in the phonetic hash to support a given file count
        while keeping the collision probability below a specified threshold, using the Birthday Paradox approximation.

        Args:
            file_count (int): The number of files.
            accepted_collision_prob (float): The accepted probability of a hash collision (0 < p < 1).

        Returns:
            int: The minimum number of word pairs required.

        Raises:
            ValueError: If file_count is less than 1 or accepted_collision_prob is not between 0 and 1 (exclusive).
        """
        if file_count < 1:
            raise ValueError("file_count must be at least 1.")
        if not (0 < accepted_collision_prob < 1):
            raise ValueError("accepted_collision_prob must be between 0 and 1 (exclusive).")

        adjective_count = len(self.adjectives)
        word_count = len(self.words_list)
        total_combinations = adjective_count * word_count

        # Calculate the required hash space H using the Birthday Paradox approximation:
        required_H = (file_count ** 2) / (-2 * math.log(1 - accepted_collision_prob))

        # Given that each word pair contributes a factor of 'total_combinations' to the hash space,
        min_word_pairs = math.ceil(math.log(required_H) / math.log(total_combinations))

        return min_word_pairs
