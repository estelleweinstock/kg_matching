from sentence_transformers import SentenceTransformer, util

def calc_acc(lefts, rights):
    output = util.semantic_search(lefts, rights)
    matches = []
    matches_count = 0
    for i in range(0, len(output)):
      matches.append([i, output[i][0]['corpus_id'] ])
      if i == output[i][0]['corpus_id']:
        matches_count = matches_count + 1
    return matches_count / len (output), matches
