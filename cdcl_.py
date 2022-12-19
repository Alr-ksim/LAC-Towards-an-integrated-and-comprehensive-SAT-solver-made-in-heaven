import bisect
from tqdm import tqdm
import time, datetime
import numpy as np
import math
import random

class cdcl_restart():
    def __init__(self, sentence, num_vars, taskname, decide_method, restart_method):
        self.sentence = sentence
        self.num_vars = num_vars

        self.taskname = taskname
        self.taskname_write = taskname.split('/')[-1] + '_' + decide_method + '_' + restart_method
        self.decide_method = decide_method
        self.restart_method = restart_method
        self.vsids_scores = None 
        self.num_conflicts = 0

        # *************************************
        # variable for restart
        # ************************************
        self.conflict_threshold = 10
        self.time_threshold = 10
        self.num_arms = 3
        self.emp_means = np.zeros(self.num_arms)
        self.num_pulls = np.ones(self.num_arms)
        self.name = ['VSIDS', 'CHB', 'LRB']

    def solve(self):
        self.start_time = time.time()

        if self.restart_method == 'Nothing':
            solver = cdcl(self.sentence, self.num_vars, self.taskname, self.decide_method, self.restart_method, self.conflict_threshold, self.vsids_scores)
            Flag, num_decisions, sentence_tmp, vsids_scores_tmp, res, num_conflicts = solver.solve()
            self.num_conflicts += num_conflicts
            self.end_time = time.time()
            self.log()
            return res

        elif self.restart_method == 'UCB':
            for t in range(self.time_threshold):
                if t < self.num_arms:
                    arm = t
                    solver = cdcl(self.sentence, self.num_vars, self.taskname, self.name[arm], self.restart_method, self.conflict_threshold, self.vsids_scores)
                    Flag, num_decisions, sentence_tmp, vsids_scores_tmp, res, num_conflicts = solver.solve()
                    self.num_conflicts += num_conflicts
                    if Flag:
                        self.end_time = time.time()
                        self.log()
                        return res
                    else:
                        self.sentence = sentence_tmp
                        self.vsids_scores = vsids_scores_tmp
                        self.emp_means[arm] = math.log(num_decisions, 2)/len(res)
                    continue

                self.choice = np.zeros(self.num_arms)
                for i in range(self.num_arms):
                    self.choice[i] = self.emp_means[i] + math.sqrt(4*math.log(t)/self.num_pulls[i])
                arm = np.argmax(self.choice)
                # print(self.choice)
                # print(self.name[arm])
                solver = cdcl(self.sentence, self.num_vars, self.taskname, self.name[arm], self.restart_method, self.conflict_threshold, self.vsids_scores)
                Flag, num_decisions, sentence_tmp, vsids_scores_tmp, res, num_conflicts = solver.solve()
                self.num_conflicts += num_conflicts
                if Flag:
                    self.end_time = time.time()
                    self.log()
                    return res
                else:
                    self.sentence = sentence_tmp
                    self.vsids_scores = vsids_scores_tmp
                    reward_tmp = math.log(num_decisions, 2)/len(res)
                    self.num_pulls[arm] += 1
                    self.emp_means[arm] = (self.emp_means[arm]*(self.num_pulls[arm]-1)+reward_tmp)/(self.num_pulls[arm])
                    # print(len(self.sentence))
            solver = cdcl(self.sentence, self.num_vars, self.taskname, self.decide_method, 'Nothing', self.conflict_threshold, self.vsids_scores)
            Flag, num_decisions, sentence_tmp, vsids_scores_tmp, res, num_conflicts = solver.solve()
            self.num_conflicts += num_conflicts
            self.end_time = time.time()
            self.log()
            return res
        elif self.restart_method == 'EXP3':
            print(1111111111111111111111)
            weight = np.ones(self.num_arms)
            possible = np.ones(self.num_arms)
            g_value = 2*self.time_threshold/3
            gamma = min(1,math.sqrt((self.num_arms*math.log(self.num_arms))/((math.e-1)*g_value)))
            for t in range(self.time_threshold):
                for i in range(self.num_arms):
                    possible[i] = (1-gamma)*weight[i]/np.sum(weight) + gamma/self.num_arms
                ra = random.uniform(0,1)
                curr_sum = 0
                curr_arm = 0
                for i in range(self.num_arms):
                    curr_sum += possible[i]
                    if ra <= curr_sum:
                        curr_arm = i
                        break
                solver = cdcl(self.sentence, self.num_vars, self.taskname, self.name[curr_arm], self.restart_method, self.conflict_threshold, self.vsids_scores)
                Flag, num_decisions, sentence_tmp, vsids_scores_tmp, res, num_conflicts = solver.solve()
                self.num_conflicts += num_conflicts
                if Flag:
                    self.end_time = time.time()
                    self.log()
                    return res
                else:
                    self.sentence = sentence_tmp
                    self.vsids_scores = vsids_scores_tmp
                    reward_tmp = math.log(num_decisions, 2)/len(res)
                    weight[curr_arm] = weight[curr_arm] * math.exp(gamma*reward_tmp/self.num_arms)
            
            solver = cdcl(self.sentence, self.num_vars, self.taskname, self.decide_method, 'Nothing', self.conflict_threshold, self.vsids_scores)
            Flag, num_decisions, sentence_tmp, vsids_scores_tmp, res, num_conflicts = solver.solve()
            self.num_conflicts += num_conflicts
            self.end_time = time.time()
            self.log()
            return res
                
    def log(self):
        w = "Time consumed: {} with conflict {} times".format(self.end_time - self.start_time, self.num_conflicts)
        file_name = './logs/' + datetime.date.today().strftime('%m%d') + "_{}.log".format(self.taskname_write)
        t0 = datetime.datetime.now().strftime('%H:%M:%S')
        info = "{} : {}".format(t0, w)
        print('*' * 30)
        print(info)
        print('*' * 30)
        print(file_name)
        with open(file_name, 'a') as f:
            f.write(info + '\n')

class cdcl():
    def __init__(self, sentence, num_vars, taskname, decide_method, restart_method, conflict_threshold, vsids_scores):
        self.sentence = sentence
        self.num_vars = num_vars

        self.init_score_func_dict = {
            'VSIDS': self.init_vsids_scores,
            'CHB': self.init_CHB_scores,
            'LRB': self.init_LRB_scores
        }

        self.update_score_func_dict = {
            'VSIDS': self.update_vsids_scores,
            'CHB': self.update_chb_scores,
            'LRB': self.update_lrb_scores
        }
        self.decide_func_dict = {
            'VSIDS': self.decide_vsids,
            'CHB': self.decide_chb,
            'LRB': self.decide_lrb
        }

        self.taskname = taskname.split('/')[-1] + '_' + decide_method + '_' + restart_method
        self.decide_method = decide_method
        self.vsids_scores = vsids_scores

        # *************************************
        # variable for restart
        # ************************************
        self.conflict_threshold = conflict_threshold
        self.restart_method = restart_method
        self.num_decisions = 0

        # *************************************
        # variable for both CHB and LRB
        # ************************************
        self.alpha = None

        # ************************************
        # variables that CHB algorithm needs
        # ***********************************
        self.CHB_scores = None
        self.num_conflicts = 0
        self.plays = None
        self.lastConflict = None
        self.multiplier = None
        self.UIP = []

        # ************************************
        # variables that LRB algorithm needs
        # ************************************
        self.LRB_scores = None
        self.learntCounter = None
        self.assigned = None
        self.participated = None
        self.reasoned = None

        self.init_score()
        self.assignments = []
        self.decided_idx = []
        self.assign_map = [0] * (2 * num_vars + 1)  # A list to record the literal is assigned or not.
        self.ante_reason = [[]] * (2 * num_vars + 1)  # A dict to record the antecedent clause of an assigned var.
        self.c2l_watch, self.l2c_watch = self.init_watch()
        self.start_time = None
        self.end_time = None

    def init_score(self):
        self.init_score_func_dict[self.decide_method]()

    def update_score(self):
        self.update_score_func_dict[self.decide_method]()

    def decide_score(self):
        return self.decide_func_dict[self.decide_method]()

    def init_vsids_scores(self):
        """Initialize variable scores for VSIDS."""
        scores = {lit: 0 for lit in range(-self.num_vars, self.num_vars + 1) if lit}
        for clause in self.sentence:
            for lit in clause:
                scores[lit] += 1
        self.vsids_scores = dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))
        # NOTE: To speed up the progress of finding the next assigned literal, I keep the list in descending order.

    # initialize CHB algorithms
    def init_CHB_scores(self):
        self.alpha = 0.4
        self.num_conflicts = 0
        self.plays = set()
        self.lastConflict = {lit: 0 for lit in range(-self.num_vars, self.num_vars + 1) if lit}
        self.CHB_scores = {lit: 0 for lit in range(-self.num_vars, self.num_vars + 1) if lit}

    # initialize LRB algorithms
    def init_LRB_scores(self):
        self.learntCounter = 0
        self.alpha = 0.4
        self.num_conflicts = 0
        self.LRB_scores = {lit: 0 for lit in range(-self.num_vars, self.num_vars + 1) if lit}
        self.assigned = {lit: 0 for lit in range(-self.num_vars, self.num_vars + 1) if lit}
        self.participated = {lit: 0 for lit in range(-self.num_vars, self.num_vars + 1) if lit}
        self.reasoned = {lit: 0 for lit in range(-self.num_vars, self.num_vars + 1) if lit}

    def update_vsids_scores(self, learned_clause, decay=0.95):
        """Update VSIDS scores."""
        for lit in self.vsids_scores:
            self.vsids_scores[lit] = self.vsids_scores[lit] * decay

        for lit in learned_clause:
            self.vsids_scores[lit] += 1

        self.vsids_scores = dict(sorted(self.vsids_scores.items(), key=lambda kv: kv[1], reverse=True))

    def update_lrb_scores(self, decay=0.95):
        """Update VSIDS scores."""
        for lit in self.LRB_scores:
            if lit not in self.assignments:
                self.LRB_scores[lit] = self.LRB_scores[lit] * decay


        self.LRB_scores = dict(sorted(self.vsids_scores.items(), key=lambda kv: kv[1], reverse=True))

    def update_chb_scores(self):
        for v in self.plays:
            reward = self.multiplier / (self.num_conflicts - self.lastConflict[v] + 1)
            self.CHB_scores[v] = (1 - self.alpha) * self.CHB_scores[v] + self.alpha * reward

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

    def decide_chb(self):
        assigned_lit = None
        for lit in self.CHB_scores.keys():
            if (self.assign_map[lit + self.num_vars]) or (self.assign_map[-lit + self.num_vars]):
                continue
            assigned_lit = lit
            break
        return assigned_lit

    def decide_lrb(self):
        assigned_lit = None
        for lit in self.LRB_scores.keys():
            if (self.assign_map[lit + self.num_vars]) or (self.assign_map[-lit + self.num_vars]):
                continue
            assigned_lit = lit
            break
        return assigned_lit

    def solve(self):
        # self.start_time = time.time()
        if self.bcp() is not None:
            return True, self.num_decisions, self.sentence, self.vsids_scores, None, self.num_conflicts
        with tqdm(total=self.num_vars) as pbar:
            assignment_prev = 0
            while (len(self.assignments) < self.num_vars):
                pbar.update(len(self.assignments) - assignment_prev)
                assignment_prev = len(self.assignments)

                assigned_lit = self.decide_score()
                self.decided_idx.append((len(self.assignments)))
                self.assignments.append(assigned_lit)
                self.num_decisions += 1 # for restart
                self.assign_map[assigned_lit + self.num_vars] = 1
                self.ante_reason[assigned_lit + self.num_vars] = []

                # Run BCP.
                conflict_ante = self.bcp()
                # ***********************************
                # for LRB algorithm
                # ***********************************
                if self.decide_method == 'LRB':
                    for v in self.assignments[self.decided_idx[-1]:]:
                        self.assigned[v] = self.learntCounter
                        self.participated[v] = 0
                        self.reasoned[v] = 0

                # ***********************************
                # for CHB algorithm
                # ***********************************
                if self.decide_method == 'CHB':
                    # record the variables before bp
                    assigned_prev = set(self.assignments)
                    # update plays
                    self.plays = {assigned_lit}
                    # set multiplier
                    self.multiplier = 1.0 if conflict_ante is not None else 0.9

                while conflict_ante is not None:
                    self.num_conflicts += 1
                    if self.restart_method != 'Nothing':
                        # print(self.num_conflicts)
                        if self.num_conflicts > self.conflict_threshold:
                            return False, self.num_decisions, self.sentence, self.vsids_scores, self.assignments, self.num_conflicts

                    # Learn conflict.
                    backtrack_level, learned_clause = self.analyse_conflict(conflict_ante)
                    # *******************************
                    # for CHB algorithm
                    # *******************************
                    if self.decide_method == 'CHB':
                        if self.alpha > 0.06:
                            self.alpha -= 1e-6
                        tail = self.decided_idx[backtrack_level]
                        c = self.assignments[tail:len(self.assignments)]
                        for v in c:
                            self.lastConflict[v] = self.num_conflicts
                        self.plays = set(self.UIP)
                    # ********************************
                    # for LRB algorithm
                    # ********************************
                    if self.decide_method == 'LRB':
                        self.learntCounter += 1
                        for v in set(learned_clause).union(set(conflict_ante)):
                            self.participated[v] += 1
                        for v in set(self.UIP).difference(set(learned_clause)):
                            self.reasoned[v] += 1

                        if self.alpha > 0.06:
                            self.alpha -= 1e-6

                    self.add_learned_clause(learned_clause)

                    if self.decide_method == 'VSIDS':
                        # Update VSIDS scores.
                        self.update_vsids_scores(learned_clause)

                    # Backtrack.
                    if backtrack_level < 0:
                        return True, self.num_decisions, self.sentence, self.vsids_scores, None, self.num_conflicts
                    self.backtrack(backtrack_level)

                    # Propagate watch.
                    conflict_ante = self.bcp(new_clause_tag=1)

            # *******************************
            # for CHB algorithm
            # *******************************
            if self.decide_method == 'CHB':
                # calculate variables played
                self.plays.union(assigned_prev.difference(set(self.assignments)))
                # update CHB_value
                self.update_chb_scores()

        # self.end_time = time.time()
        # self.log()
        return True, self.num_decisions, self.sentence, self.vsids_scores, self.assignments, self.num_conflicts  # indicate SAT

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
        self.UIP = [last_lit]
        ante = self.ante_reason[last_lit + self.num_vars]
        key_var = abs(last_lit)
        learned_clause = self.resolve(learned_clause, ante, key_var)

        while not self.unit_at_level(learned_clause, conflict_level):
            last_lit = self.last_assigned_lit(set(learned_clause), conflict_level)
            self.UIP.append(last_lit)
            ante = self.ante_reason[last_lit + self.num_vars]
            key_var = abs(last_lit)
            learned_clause = self.resolve(learned_clause, ante, key_var)

        backtrack_level = self.track_least(set(learned_clause))

        return backtrack_level, learned_clause

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
        # **********************************
        # for LRB algorithm
        # **********************************
        if self.decide_method == 'LRB':
            for v in self.assignments[tail:n]:
                interval = self.learntCounter - self.assigned[v]
                if interval > 0:
                    r = self.participated[v] / interval
                    rsr = self.reasoned[v] / interval
                    self.LRB_scores[v] = (1 - self.alpha) * self.LRB_scores[v] + self.alpha * (r + rsr)
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

    # def log(self):
    #     w = "Time consumed: {} with conflict {} times".format(self.end_time - self.start_time, self.num_conflicts)
    #     file_name = './logs/' + datetime.date.today().strftime('%m%d') + "_{}.log".format(self.taskname)
    #     t0 = datetime.datetime.now().strftime('%H:%M:%S')
    #     info = "{} : {}".format(t0, w)
    #     print('*' * 30)
    #     print(info)
    #     print('*' * 30)
    #     with open(file_name, 'a') as f:
    #         f.write(info + '\n')
