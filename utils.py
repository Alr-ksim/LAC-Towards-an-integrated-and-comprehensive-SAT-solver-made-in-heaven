def read_cnf(fp):
    sentence = []

    for line in fp:
        if line.startswith('c'):
            continue
        if line.startswith('p'):
            line = line.split()
            num_vars, num_clauses = int(line[2]), int(line[3])
        else:
            line = line.split()
            clause = [int(x) for x in line[:-1]]
            sentence.append(clause)

    assert len(sentence) == num_clauses

    return sentence, num_vars
