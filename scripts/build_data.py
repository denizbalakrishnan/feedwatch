"""build_data.py: makes the small real-data file used by the "Real people" panel.

It reads the raw KuaiRec download, which is not kept in this repository, and
writes data/kuairec_shown.json: for 1,000 real people and 800 real videos,
one bit per pair saying whether the Kuaishou app showed that person that
video. No captions, no user details and no original ID numbers are copied.

Usage:   python scripts/build_data.py PATH/TO/KuaiRec.zip
Source:  KuaiRec, Gao et al., CIKM 2022, https://kuairec.com (CC BY-SA 4.0)
"""
import base64
import csv
import io
import json
import random
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

# Ten categories with a clear subject, and our own English names for them.
CATEGORIES = {
    '舞蹈': 'Dance', '运动': 'Sports', '美食': 'Food', '时尚': 'Fashion', '喜剧': 'Comedy',
    '宠物': 'Pets', '艺术': 'Art', '亲子': 'Parenting', '旅游': 'Travel', '二次元': 'Anime',
}
VIDEOS_PER_CATEGORY = 80
MIN_VIEWERS = 300       # a video must have been shown to at least this many people
USER_COUNT = 1000
MIN_VIDEOS_SEEN = 40    # a person must have been shown at least this many of the chosen videos
SEED = 2026


def open_table(archive, name):
    """Opens one CSV file inside the zip as text, without unpacking the zip."""
    member = next(path for path in archive.namelist() if path.endswith(name))
    return io.TextIOWrapper(archive.open(member), encoding='utf-8-sig', newline='')


def read_categories(archive):
    """Returns {video id: English category name} for videos in the ten chosen categories."""
    labels = {}
    with open_table(archive, 'kuairec_caption_category.csv') as table:
        for row in csv.DictReader(table):
            name = CATEGORIES.get(row['first_level_category_name'])
            if name and row['video_id'].isdigit():
                labels[int(row['video_id'])] = name
    return labels


def read_viewers(archive, labels):
    """Returns {video id: set of user ids who were shown it}, from the 12.5 million rows of the big matrix."""
    viewers = defaultdict(set)
    with open_table(archive, 'big_matrix.csv') as table:
        header = table.readline().strip().split(',')
        user_column, video_column = header.index('user_id'), header.index('video_id')
        for line in table:
            parts = line.split(',')
            video = int(parts[video_column])
            if video in labels:
                viewers[video].add(int(parts[user_column]))
    return viewers


def main():
    archive = zipfile.ZipFile(sys.argv[1])
    labels = read_categories(archive)
    viewers = read_viewers(archive, labels)

    # The most-shown videos of each category, up to the limit.
    videos, taken = [], Counter()
    for video in sorted(viewers, key=lambda v: (-len(viewers[v]), v)):
        if len(viewers[video]) >= MIN_VIEWERS and taken[labels[video]] < VIDEOS_PER_CATEGORY:
            videos.append(video)
            taken[labels[video]] += 1

    # People who were shown enough of those videos, then a repeatable random 1,000 of them.
    seen_count = Counter(user for video in videos for user in viewers[video])
    users = sorted(user for user, count in seen_count.items() if count >= MIN_VIDEOS_SEEN)
    random.Random(SEED).shuffle(users)
    users = users[:USER_COUNT]

    # One bit per (person, video), row by row; bit 0 of each byte comes first.
    bits = bytearray((len(users) * len(videos) + 7) // 8)
    shown = 0
    for row, user in enumerate(users):
        for column, video in enumerate(videos):
            if user in viewers[video]:
                position = row * len(videos) + column
                bits[position >> 3] |= 1 << (position & 7)
                shown += 1

    names = list(CATEGORIES.values())
    output = {
        'source': 'KuaiRec (Gao et al., CIKM 2022), https://kuairec.com',
        'licence': 'CC BY-SA 4.0',
        'userCount': len(users),
        'videoCount': len(videos),
        'categories': names,
        'videoCategory': [names.index(labels[video]) for video in videos],
        'shownBits': base64.b64encode(bytes(bits)).decode('ascii'),
    }
    target = Path(__file__).resolve().parent.parent / 'data' / 'kuairec_shown.json'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(output), encoding='utf-8')
    print(f'{len(users)} people x {len(videos)} videos, {dict(taken)}')
    print(f'{shown / (len(users) * len(videos)):.1%} of pairs were shown; wrote {target} ({target.stat().st_size / 1024:.0f} kB)')


if __name__ == '__main__':
    main()
