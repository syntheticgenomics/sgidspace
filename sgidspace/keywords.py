import re
import glob
import gzip
import json
import argparse

from signal import signal, SIG_DFL, SIGPIPE
signal(SIGPIPE, SIG_DFL)

string_subs = [
    ['/', ' '],
    ['|', ''],
    [':"', ''],
    [':', '-'],
    ['(P)', ''],
    [' binding', '-binding'],
    [' dependent', '-dependent'],
    ['type ', 'type-'],
    ['class ', 'class-'],
    [' like', '-like'],
    ['group ', 'group-'],
    [')', ' '],
    ['(', ' '],
    ['/', ' '],
    [']', ''],
    ['[', ''],
    ['--', ' '],
    ['*', ''],
    ['co-', 'co'],
    ['protein,', ''],
    ['S. cerevisiae', ''],
]

clear_latin_subs = [
    ['ii', '2'],
    ['iii', '3'],
    ['iiii', '4'],
    ['vii', '7'],
    ['viii', '8'],
    ['viiii', '9'],
]

ambiguous_latin_subs = [
    ['i', '1'],
    ['iv', '4'],
    ['v', '5'],
    ['vi', '6'],
]

trailing_filter = ',.-\n'

keyword_filter = set([
    '', 'and', 'of', 'the', 'is', 'in', 'with', 'to', 'for', 'or', ':', '@', '+', '&', '-', '#',
    '##', '=', '>', ',', '\'', 'ec-', 'protein', 'family', 'containing', 'domain-containing',
    'component', 'system', 'type', 'putative', 'region', 'predicted', 'Drosophila', 'cerevisiae',
    'elegans', 'yeast', 'fly', 'schizosaccharomyces', 'pombe', 'eurofung', 'b.subtilis'
])

phrase_filter = set([''])

regex_filter = [
    re.compile(regex)
    for regex in [
        'duf[0-9]+', 'u*pf[0-9][0-9][0-9]+', 'afua_', 'cog[0-9][0-9][0-9]+', 'clone-', '[0-9]aa*',
        '[0-9]-*kda', 's[0-9][0-9]*h.*_[a-z][0-9].*', 'zgc-[0-9][0-9]*', '.*_ortholog', 'homolog',
        '.*ncharac.*'
    ]
]


def extract_keywords(desc, phrase=False):
    """
    if phrase is False, a set of words will be returned
    if phrase is True, a list of words will be returned in the same order that they were provided
    """
    # Remove spaces around dashes
    desc = re.sub(' *- *', '-', desc)

    # Remove kda entries
    desc = re.sub('[0-9]+ *kda', '', desc, flags=re.IGNORECASE)

    # Perform string substitutions
    for ss in string_subs:
        desc = re.sub(re.escape(ss[0]), ss[1], desc, flags=re.IGNORECASE)

    # Remove region entries
    desc = re.sub('chromosome.[0-9]*', '', desc, flags=re.IGNORECASE)
    desc = re.sub('supercontig.[0-9]*', '', desc, flags=re.IGNORECASE)
    desc = re.sub('contig.[0-9]*', '', desc, flags=re.IGNORECASE)
    desc = re.sub('scaffold.[0-9]*', '', desc, flags=re.IGNORECASE)

    # Split by spaces
    words = desc.split(' ')

    # Remove '...' entries
    words = [w for w in words if not w.endswith('...')]

    # Lowercase conversion
    words = [w.lower() for w in words]

    # Remove regex entries
    words = [w for w in words if not any(regex.match(w) for regex in regex_filter)]

    # Latin conversions (clear cases)
    for wi in xrange(len(words)):
        for ss in clear_latin_subs:
            words[wi] = re.sub(re.escape(ss[0]), ss[1], words[wi])

    # Latin conversions (ambiguous cases)
    for wi in xrange(len(words)):
        for ss in ambiguous_latin_subs:
            if words[wi] == ss[0]:
                words[wi] = ss[1]
                break

    # Remove leading and trailing characters
    words = [w.strip(trailing_filter) for w in words]

    # # Word filter
    # if phrase:
    #     words = [w for w in words if w not in phrase_filter]
    # else:
    words = [w for w in words if w not in keyword_filter]

    if phrase:
        return words
    else:
        return list(set(words))


def main():

    parser = argparse.ArgumentParser(description='Extract keywords or phrases from data files.')
    parser.add_argument('data', help="A data directory containing processed data files")
    parser.add_argument(
        '-p', '--phrase', action='store_true', help="Return phrases rather than keywords"
    )

    args = parser.parse_args()

    filenames = glob.glob(args.data + '/*.json.gz')

    for filename in filenames:
        data_handle = gzip.GzipFile(filename, 'rb')
        for data in data_handle:
            if isinstance(data, bytes):
                data = json.loads(data.decode('utf-8'))
            else:
                data = json.loads(data)
            print(
                data['translation_description'] + '\t-->\t' + '\t'.join(
                    extract_keywords(data['translation_description'], phrase=args.phrase)
                )
            )


if __name__ == '__main__':
    main()
