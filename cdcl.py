import bisect


def find_next(clause, another, vars_num, assign_map):
    # NOTE: a tool function to update the c2l_watch when necessary.
    for lit in clause:
        if (another is not None) and (lit == another):
            continue
        if not (assign_map[-lit + vars_num]):
            return lit
    return None


def test_sat(assignment, key_list, vars_num, assign_map):
    # NOTE: a tool function designed to test a clause is satisfied or not.
    if (key_list[0] is not None) and (key_list[1] is not None):
        return 1
    if key_list[0] is None:
        key_list[0] = key_list[1]
        key_list[1] = None
    if key_list[0] is None:
        return 0
    if assign_map[key_list[0] + vars_num] == 1:
        return 1
    return 0


def init_bcp(sentence, assignment, c2l_watch, l2c_watch, ante_reason, vars_num, assign_map):
    # NOTE: a tool function designed to deal with the initial BCP.
    for clause in sentence:
        if len(clause) == 1:
            assignment.append(clause[0])
            assign_map[clause[0] + vars_num] = 1
            ante_reason[clause[0] + vars_num] = clause
            res = bcp(sentence, assignment, c2l_watch, l2c_watch, ante_reason, vars_num, assign_map)
            if res is not None:
                return res
    return None


def bcp(sentence, assignment, c2l_watch, l2c_watch, ante_reason, vars_num, assign_map, new_clause_tag=0):
    # NOTE: to speed up the BCP, add ante_reason, num_vars, assign_map as extra parameters. The tag new_clause_tag
    # is designed specially for the case that a new clause is added.
    """Propagate unit clauses with watched literals."""

    """ YOUR CODE HERE """
    n = len(sentence)
    new_assign = 0
    if not assignment:
        return init_bcp(sentence, assignment, c2l_watch, l2c_watch, ante_reason, vars_num, assign_map)
    if assignment:
        new_assign = assignment[-1]

    ranges = l2c_watch[-new_assign]
    if new_clause_tag:
        if len(c2l_watch[n-1]) == 1:
            key = c2l_watch[n-1][0]
            assignment.append(key)
            assign_map[key + vars_num] = 1
            ante_reason[key + vars_num] = sentence[n-1]
            return bcp(sentence, assignment, c2l_watch, l2c_watch, ante_reason, vars_num, assign_map)
        ranges = [n-1]
    for i in ranges:
        if len(c2l_watch[i]) < 2:
            continue
        mini_cl = c2l_watch[i]
        for j in range(0, 2):
            if (mini_cl[j] is None) or (assign_map[-mini_cl[j] + vars_num] == 1):
                c2l_watch[i][j] = find_next(sentence[i], mini_cl[j ^ 1], vars_num, assign_map)
        if test_sat(assignment, c2l_watch[i], vars_num, assign_map):
            continue
        if c2l_watch[i][0] is not None:
            key = c2l_watch[i][0]
            assignment.append(key)
            assign_map[key + vars_num] = 1
            ante_reason[key + vars_num] = sentence[i]
            res = bcp(sentence, assignment, c2l_watch, l2c_watch, ante_reason, vars_num, assign_map)
            if res is not None:
                return res
        else:
            return sentence[i]

    return None  # indicate no conflict; other return the antecedent of the conflict


def init_vsids_scores(sentence, num_vars):
    """Initialize variable scores for VSIDS."""

    """ YOUR CODE HERE """
    scores = {lit: 0 for lit in range(-num_vars, num_vars + 1) if lit}
    for clause in sentence:
        for lit in clause:
            scores[lit] += 1
    scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    # NOTE: To speed up the progress of finding the next assigned literal, I keep the list in descending order.
    return dict(scores)


def decide_vsids(assignment, vsids_scores, vars_num, assign_map):
    """Decide which variable to assign and whether to assign True or False."""
    assigned_lit = None
    """ YOUR CODE HERE """
    # NOTE: As the vsids_scores dict is in descending order, we will just find the first unassigned literal.
    for lit in vsids_scores.keys():
        if (assign_map[lit + vars_num]) or (assign_map[-lit + vars_num]):
            continue
        assigned_lit = lit
        break
    return assigned_lit


def update_vsids_scores(vsids_scores, learned_clause, decay=0.95):
    """Update VSIDS scores."""
    for lit in vsids_scores:
        vsids_scores[lit] = vsids_scores[lit] * decay

    for lit in learned_clause:
        vsids_scores[lit] += 1

    scores = sorted(vsids_scores.items(), key=lambda kv: kv[1], reverse=True)
    # NOTE: To speed up the progress of finding the next assigned literal, I keep the list in descending order.
    return dict(scores)

def init_watch(sentence, num_vars):
    """Initialize the watched literal data structure."""

    """ YOUR CODE HERE """
    n = len(sentence)
    c2l_watch = {i: [lit for lit in sentence[i]][:2] for i in range(0, n)}  # clause -> literal
    l2c_watch = {lit: [] for lit in range(-num_vars, num_vars + 1) if lit}  # literal -> watch

    for i in range(0, n):
        for lit in sentence[i]:
            l2c_watch[lit].append(i)

    return c2l_watch, l2c_watch


def last_assigned_lit(assignment, decided_idxs, clause, level):
    # NOTE: a tool function to find the last assigned literal in the clause.
    if level > len(decided_idxs):
        return 0
    decided_idxs.append(len(assignment))
    head = decided_idxs[level-1]
    tail = decided_idxs[level]
    del decided_idxs[-1]

    for i in range(tail-1, head-1, -1):
        if -assignment[i] in clause:
            return assignment[i]
    return 0


def unit_at_level(assignment, decided_idxs, clause, level):
    # NOTE: a tool function to check whether there is only one literal in the clause with a given level.
    if level > len(decided_idxs):
        return 0
    decided_idxs.append(len(assignment))
    level_lit = set(assignment[decided_idxs[level - 1]: decided_idxs[level]])
    del decided_idxs[-1]

    bingo = [lit for lit in clause if -lit in level_lit]
    if len(bingo) == 1:
        return 1
    return 0


def resolve(cl1, cl2, var):
    # NOTE: a tool function to resolve two clauses.
    cl = set(cl1 + cl2)
    if var in cl:
        cl.remove(var)
    if -var in cl:
        cl.remove(-var)
    return list(cl)


def track_least(assignment, decided_idxs, clause):
    # NOTE: a tool function to find the minimum level of the literals in the clause.
    level = 0
    if len(clause) == 1:
        return level
    for i in range(0, len(assignment)):
        if (-assignment[i]) not in clause:
            continue
        return bisect.bisect(decided_idxs, i)
    return None


def analyze_conflict(assignment, decided_idxs, conflict_ante, vars_num, ante_reason):
    # NOTE: To speed up the analysis, I add vars_num and ante_reason as extra parameters.
    """Analyze the conflict with first-UIP clause learning."""
    backtrack_level, learned_clause = None, []

    """ YOUR CODE HERE """
    learned_clause = conflict_ante
    conflict_level = len(decided_idxs)
    if conflict_level == 0:
        return -1, []
    last_lit = last_assigned_lit(assignment, decided_idxs, learned_clause, conflict_level)
    ante = ante_reason[last_lit + vars_num]
    key_var = abs(last_lit)
    learned_clause = resolve(learned_clause, ante, key_var)

    while not unit_at_level(assignment, decided_idxs, learned_clause, conflict_level):
        last_lit = last_assigned_lit(assignment, decided_idxs, set(learned_clause), conflict_level)
        ante = ante_reason[last_lit + vars_num]
        key_var = abs(last_lit)
        learned_clause = resolve(learned_clause, ante, key_var)

    backtrack_level = track_least(assignment, decided_idxs, set(learned_clause))

    return backtrack_level, learned_clause


def backtrack(assignment, decided_idxs, ante_reason, level, vars_num, assign_map):
    """Backtrack by deleting assigned variables."""

    """ YOUR CODE HERE """
    tail = decided_idxs[level]
    n = len(assignment)
    for i in range(tail, n):
        assign_map[assignment[i] + vars_num] = 0
        ante_reason[assignment[i] + vars_num] = []
    del assignment[tail:n]
    del decided_idxs[level:n]


def add_learned_clause(sentence, learned_clause, c2l_watch, l2c_watch):
    """Add learned clause to the sentence and update watch."""

    """ YOUR CODE HERE """
    idx = len(sentence)
    sentence.append(learned_clause)
    c2l_watch.update({idx: [lit for lit in learned_clause][:2]})
    for lit in learned_clause:
        l2c_watch[lit].append(idx)


def test_assign(sentence, assignment):
    for clause in sentence:
        flag = 0
        for lit in clause:
            if lit in assignment:
                flag = 1
                break
        if flag == 0:
            return 0
    return 1


def cdcl(sentence, num_vars):
    """Run a CDCL solver for the SAT problem.

    To simplify the use of data structures, `sentence` is a list of lists where each list
    is a clause. Each clause is a list of literals, where a literal is a signed integer.
    `assignment` is also a list of literals in the order of their assignment.
    """
    # Initialize some data structures.
    vsids_scores = init_vsids_scores(sentence, num_vars)
    assignment, decided_idxs = [], []
    assign_map = [0] * (2 * num_vars + 1)       # NOTE: a list to record the literal is assigned or not.
    ante_reason = [[]] * (2 * num_vars + 1)     # NOTE: a dict to record the antecedent clause of an assigned var.
    c2l_watch, l2c_watch = init_watch(sentence, num_vars)

    # Run BCP.
    if bcp(sentence, assignment, c2l_watch, l2c_watch, ante_reason, num_vars, assign_map) is not None:
        return None  # indicate UNSAT

    # Main loop.
    while len(assignment) < num_vars:
        assigned_lit = decide_vsids(assignment, vsids_scores, num_vars, assign_map)
        decided_idxs.append(len(assignment))
        assignment.append(assigned_lit)
        assign_map[assigned_lit + num_vars] = 1
        ante_reason[assigned_lit + num_vars] = []

        # Run BCP.
        conflict_ante = bcp(sentence, assignment, c2l_watch, l2c_watch, ante_reason, num_vars, assign_map)

        while conflict_ante is not None:
            # Learn conflict.
            backtrack_level, learned_clause = analyze_conflict(assignment, decided_idxs, conflict_ante, num_vars, ante_reason)
            add_learned_clause(sentence, learned_clause, c2l_watch, l2c_watch)

            # Update VSIDS scores.
            vsids_scores = update_vsids_scores(vsids_scores, learned_clause)

            # Backtrack.
            if backtrack_level < 0:
                return None
            backtrack(assignment, decided_idxs, ante_reason, backtrack_level, num_vars, assign_map)

            # Propagate watch.
            conflict_ante = bcp(sentence, assignment, c2l_watch, l2c_watch, ante_reason, num_vars, assign_map, 1)

    return assignment  # indicate SAT
