import bisect
from tqdm import tqdm
import time, datetime


class cdcl():
    def __init__(self, sentence, num_vars,taskname):
        self.sentence = sentence
        self.num_vars = num_vars
        self.vsids_scores = self.init_vsids_scores()
        self.assignments = []
        self.decided_idx = []
        self.assign_map = [0] * (2 * num_vars + 1)  # A list to record the literal is assigned or not.
        self.ante_reason = [[]] * (2 * num_vars + 1)  # A dict to record the antecedent clause of an assigned var.
        self.c2l_watch, self.l2c_watch = self.init_watch()
        self.start_time = None
        self.end_time = None
        self.num_conflicts = 0
        self.taskname = taskname.split('/')[-1]

    def update_vsids_scores(self, learned_clause, decay=0.95):
        """Update VSIDS scores."""
        for lit in self.vsids_scores:
            self.vsids_scores[lit] = self.vsids_scores[lit] * decay

        for lit in learned_clause:
            self.vsids_scores[lit] += 1

        self.vsids_scores = dict(sorted(self.vsids_scores.items(), key=lambda kv: kv[1], reverse=True))

    def solve(self):
        self.start_time = time.time()
        if self.bcp() is not None:
            return None
        with tqdm(total=self.num_vars) as pbar:
            assignment_prev = 0
            while (len(self.assignments) < self.num_vars):
                pbar.update(len(self.assignments) - assignment_prev)
                assignment_prev = len(self.assignments)

                assigned_lit = self.decide_vsids()
                self.decided_idx.append((len(self.assignments)))
                self.assignments.append(assigned_lit)
                self.assign_map[assigned_lit + self.num_vars] = 1
                self.ante_reason[assigned_lit + self.num_vars] = []

                # Run BCP.
                conflict_ante = self.bcp()

                while conflict_ante is not None:
                    self.num_conflicts += 1
                    # Learn conflict.
                    backtrack_level, learned_clause = self.analyse_conflict(conflict_ante)
                    self.add_learned_clause(learned_clause)

                    # Update VSIDS scores.
                    self.update_vsids_scores(learned_clause)

                    # Backtrack.
                    if backtrack_level < 0:
                        return None
                    self.backtrack(backtrack_level)

                    # Propagate watch.
                    conflict_ante = self.bcp(new_clause_tag=1)
        self.end_time = time.time()
        self.log()
        return self.assignments  # indicate SAT

    def add_learned_clause(self, learned_clause):
        """Add learned clause to the sentence and update watch."""
        idx = len(self.sentence)
        self.sentence.append(learned_clause)
        self.c2l_watch.update({idx: [lit for lit in learned_clause][:2]})
        for lit in learned_clause:
            self.l2c_watch[lit].append(idx)

    def last_assigned_lit(self, clause, level):
        # NOTE: a tool function to find the last assigned literal in the clause.
        if level > len(self.decided_idx):
            return 0
        self.decided_idx.append(len(self.assignments))
        head = self.decided_idx[level - 1]
        tail = self.decided_idx[level]
        del self.decided_idx[-1]

        for i in range(tail - 1, head - 1, -1):
            if -self.assignments[i] in clause:
                return self.assignments[i]
        return 0

    def resolve(self, cl1, cl2, var):
        # NOTE: a tool function to resolve two clauses.
        cl = set(cl1 + cl2)
        if var in cl:
            cl.remove(var)
        if -var in cl:
            cl.remove(-var)
        return list(cl)

    def unit_at_level(self, clause, level):
        # NOTE: a tool function to check whether there is only one literal in the clause with a given level.
        if level > len(self.decided_idx):
            return 0
        self.decided_idx.append(len(self.assignments))
        level_lit = set(self.assignments[self.decided_idx[level - 1]: self.decided_idx[level]])
        del self.decided_idx[-1]

        bingo = [lit for lit in clause if -lit in level_lit]
        return len(bingo) == 1

    def track_least(self, clause):
        # NOTE: a tool function to find the minimum level of the literals in the clause.
        level = 0
        if len(clause) == 1:
            return level
        for i in range(0, len(self.assignments)):
            if (-self.assignments[i]) not in clause:
                continue
            return bisect.bisect(self.decided_idx, i)
        return None

    def analyse_conflict(self, conflict_ante):
        learned_clause = conflict_ante
        conflict_level = len(self.decided_idx)
        if conflict_level == 0:
            return -1, []
        last_lit = self.last_assigned_lit(learned_clause, conflict_level)
        ante = self.ante_reason[last_lit + self.num_vars]
        key_var = abs(last_lit)
        learned_clause = self.resolve(learned_clause, ante, key_var)

        while not self.unit_at_level(learned_clause, conflict_level):
            last_lit = self.last_assigned_lit(set(learned_clause), conflict_level)
            ante = self.ante_reason[last_lit + self.num_vars]
            key_var = abs(last_lit)
            learned_clause = self.resolve(learned_clause, ante, key_var)

        backtrack_level = self.track_least(set(learned_clause))

        return backtrack_level, learned_clause

    def decide_vsids(self):
        assigned_lit = None
        # NOTE: As the vsids_scores dict is in descending order, we will just find the first unassigned literal.
        for lit in self.vsids_scores.keys():
            # 加vars_num以保证其在索引的时候>0,实在是太妙了
            if (self.assign_map[lit + self.num_vars]) or (self.assign_map[-lit + self.num_vars]):
                continue
            assigned_lit = lit
            break
        return assigned_lit

    def init_vsids_scores(self):
        """Initialize variable scores for VSIDS."""

        """ YOUR CODE HERE """
        scores = {lit: 0 for lit in range(-self.num_vars, self.num_vars + 1) if lit}
        for clause in self.sentence:
            for lit in clause:
                scores[lit] += 1
        scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        # NOTE: To speed up the progress of finding the next assigned literal, I keep the list in descending order.
        return dict(scores)

    def init_watch(self):
        n = len(self.sentence)
        c2l_watch = {i: [lit for lit in self.sentence[i]][:2] for i in range(0, n)}  # clause -> literal
        # 没有变元0,所以加入if lit特判
        l2c_watch = {lit: [] for lit in range(-self.num_vars, self.num_vars + 1) if lit}  # literal -> watch

        for i in range(0, n):
            for lit in self.sentence[i]:
                l2c_watch[lit].append(i)

        return c2l_watch, l2c_watch

    def init_bcp(self):
        for clause in self.sentence:
            if len(clause) == 1:
                self.assignments.append(clause[0])
                self.assign_map[clause[0] + self.num_vars] = 1
                self.ante_reason[clause[0] + self.num_vars] = clause
                res = self.bcp()
                if res is not None:
                    return res
        return None

    def find_next(self, clause, another):
        for lit in clause:
            if (another is not None) and (lit == another):
                continue
            if not (self.assign_map[-lit + self.num_vars]):
                return lit
        return None

    def test_sat(self, key_list):
        # NOTE: a tool function designed to test a clause is satisfied or not.
        if (key_list[0] is not None) and (key_list[1] is not None):
            return 1
        if key_list[0] is None:
            key_list[0] = key_list[1]
            key_list[1] = None
        if key_list[0] is None:
            return 0
        if self.assign_map[key_list[0] + self.num_vars] == 1:
            return 1
        return 0

    def backtrack(self, level):
        """Backtrack by deleting assigned variables."""

        """ YOUR CODE HERE """
        tail = self.decided_idx[level]
        n = len(self.assignments)
        for i in range(tail, n):
            self.assign_map[self.assignments[i] + self.num_vars] = 0
            self.ante_reason[self.assignments[i] + self.num_vars] = []
        del self.assignments[tail:n]
        del self.decided_idx[level:n]

    def bcp(self, new_clause_tag=0):
        # NOTE: to speed up the BCP, add ante_reason, num_vars, assign_map as extra parameters. The tag new_clause_tag
        # is designed specially for the case that a new clause is added.
        n = len(self.sentence)
        new_assign = 0
        if not self.assignments:
            return self.init_bcp()
        if self.assignments:
            new_assign = self.assignments[-1]

        ranges = self.l2c_watch[-new_assign]
        if new_clause_tag:
            if len(self.c2l_watch[n - 1]) == 1:
                key = self.c2l_watch[n - 1][0]
                self.assignments.append(key)
                self.assign_map[key + self.num_vars] = 1
                self.ante_reason[key + self.num_vars] = self.sentence[n - 1]
                return self.bcp()
            ranges = [n - 1]
        for i in ranges:
            if len(self.c2l_watch[i]) < 2:
                continue
            mini_cl = self.c2l_watch[i]
            for j in range(0, 2):
                if (mini_cl[j] is None) or (self.assign_map[-mini_cl[j] + self.num_vars] == 1):
                    self.c2l_watch[i][j] = self.find_next(self.sentence[i], mini_cl[j ^ 1])
            if self.test_sat(self.c2l_watch[i]):
                continue
            if self.c2l_watch[i][0] is not None:
                key = self.c2l_watch[i][0]
                self.assignments.append(key)
                self.assign_map[key + self.num_vars] = 1
                self.ante_reason[key + self.num_vars] = self.sentence[i]
                res = self.bcp()
                if res is not None:
                    return res
            else:
                return self.sentence[i]

        return None  # indicate no conflict; other return the antecedent of the conflict

    def log(self):
        w = "Time consumed: {} with conflict {} times".format(self.end_time-self.start_time,self.num_conflicts)
        file_name = './logs/' + datetime.date.today().strftime('%m%d') + "_{}.log".format(self.taskname)
        t0 = datetime.datetime.now().strftime('%H:%M:%S')
        info = "{} : {}".format(t0, w)
        print('*' * 30)
        print(info)
        print('*' * 30)
        with open(file_name, 'a') as f:
            f.write(info + '\n')
