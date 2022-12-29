from bisect import bisect
from collections import deque
from tqdm import tqdm
import math
import time
import numpy as np
import random


class Simplify:
    def __init__(self, sentence, num_vars, decide_method, restart_method):
        self.initial = sentence
        self.sentence = [set(clause) for clause in sentence]
        self.num_vars = num_vars
        self.touched = {lit for lit in range(-num_vars, num_vars+1) if lit}
        self.added = {idx for idx in range(0, len(sentence))}
        self.strengthened = set()
        self.assigned = {lit: 0 for lit in range(-num_vars, num_vars+1) if lit}
        self.assign_list = []

        self.decide_method = decide_method
        self.restart_method = restart_method

        self.mod = 64
        self.unsat = False
        self.deleted = {idx: 0 for idx in range(0, len(sentence))}
        self.l2c = {lit: set() for lit in range(-num_vars, num_vars+1) if lit}

        for idx, clause in enumerate(sentence):
            for lit in clause:
                self.l2c[lit].add(idx)

    def get_label(self, clause):
        label = 0
        for lit in clause:
            label |= 1 << ((lit + self.num_vars) % self.mod)
        return label

    def subset(self, a, b):
        la = self.get_label(a)
        lb = self.get_label(b)
        if la & (~lb):
            return False
        for lit in a:
            if lit not in b:
                return False
        return True

    def find_subsumed(self, clause):
        if not clause:
            return
        cl = clause.copy()
        plist = {lit: len(self.l2c[lit]) for lit in cl}
        p = min(plist.items(), key=lambda x: x[1])[0]
        results = []
        for i in self.l2c[p]:
            if not self.deleted[i] and len(cl) < len(self.sentence[i]) \
                    and self.subset(cl, self.sentence[i]):
                results.append(i)
        return results

    def propagate_toplevel(self):
        for idx, clause in enumerate(self.sentence):
            if self.deleted[idx] or len(clause) != 1:
                continue
            lit = list(clause)[0]
            if self.assigned[-lit]:
                self.unsat = True
                return
            if not self.assigned[lit]:
                self.assigned[lit] = 1
                self.assign_list.append(lit)
                for i in self.l2c[lit]:
                    self.deleted[i] = lit  # 1 or lit
                    self.touch_add(self.sentence[i])
                sets = self.l2c[-lit].copy()
                for i in sets:
                    self.sentence[i].remove(-lit)
                    if not self.sentence[i]:
                        self.unsat = True
                        return
                    self.l2c[-lit].remove(i)
                    self.strengthened.add(i)
                    self.touch_add(self.sentence[i])

    def subsumed(self, clause):
        res = self.find_subsumed(clause)
        if res:
            for idx in res:
                self.deleted[idx] = 1
                self.touch_add(self.sentence[idx])

    def touch_add(self, clause):
        for lit in clause:
            if not self.assigned[lit]:
                self.touched.add(lit)

    def satisfy(self, clause):
        for lit in clause:
            assert self.assigned[lit] == 1 or self.assigned[-lit] == 1
            if self.assigned[lit]:
                return True
        return False

    def self_subsumed(self, clause):
        sets = clause.copy()
        for lit in sets:
            cl = sets.copy()
            cl.remove(lit)
            cl.add(-lit)
            res = self.find_subsumed(cl)
            if not res:
                return
            for idx in res:
                self.sentence[idx].remove(-lit)
                self.l2c[-lit].remove(idx)
                self.strengthened.add(idx)
                self.touch_add(self.sentence[idx])

    @staticmethod
    def is_tautology(clause):
        pos, neg = set(), set()
        for lit in clause:
            if lit > 0: pos.add(lit)
            if lit < 0: neg.add(-lit)
        if pos.intersection(neg):
            return True
        return False

    def maybe_clause_distribute(self, lit, pos, neg):
        resolves = []
        assigns = []
        sources = []
        for idx in pos + neg:
            sources.append(self.sentence[idx])
        for pi in pos:
            for ni in neg:
                res = self.sentence[pi].union(self.sentence[ni])
                res.remove(lit)
                res.remove(-lit)
                if not self.is_tautology(res):
                    resolves.append(res)

        if len(resolves) > len(pos + neg):
            return False
        else:
            for pi in pos:
                p = self.sentence[pi].copy()
                p.remove(lit)
                if p:
                    assigns.append(p)
            if not resolves and not assigns:
                self.unsat = True
                return False
            if assigns:
                self.assigned[lit] = assigns
            else:
                self.assigned[lit] = 1
            self.assign_list.append(lit)
            for idx in pos+neg:
                self.deleted[idx] = lit
                self.touch_add(self.sentence[idx])
            for clause in resolves:
                idx = len(self.sentence)
                self.sentence.append(clause)
                self.deleted[idx] = 0
                self.added.add(idx)
                self.touch_add(clause)
                for lit in clause:
                    self.l2c[lit].add(idx)
            return True

    def maybe_eliminate(self, lit):
        if self.assigned[lit] or self.assigned[-lit]:
            return None
        pos = [idx for idx in self.l2c[lit] if not self.deleted[idx]]
        neg = [idx for idx in self.l2c[-lit] if not self.deleted[idx]]
        if len(pos) > 10 and len(neg) > 10:
            return None
        if not pos or not neg:
            return None
        if self.maybe_clause_distribute(lit, pos, neg):
            self.propagate_toplevel()

    @staticmethod
    def translate(source, dicts):
        trans = []
        for lit in source:
            if lit > 0:
                trans.append(dicts[lit])
            else:
                trans.append(-dicts[-lit])
        return trans

    def tester(self):
        for idx, clause in enumerate(self.initial):
            if not self.satisfy(clause):
                return 0
        return 1

    def solve(self):
        t0 = time.time()
        print(f'Pre-processing ......')
        for idx, clause in enumerate(self.sentence):
            if self.is_tautology(clause):
                self.deleted[idx] = 1
                self.added.remove(idx)

        while self.added:
            add_set = set()
            for idx in self.added:
                add_set = add_set.union(self.sentence[idx])
            s0 = set()
            for idx, clause in enumerate(self.sentence):
                if self.deleted[idx] or len(clause) > 2:
                    continue
                for lit in clause:
                    if lit in add_set:
                        s0.add(idx)
                        break
            flag = 1
            while flag or self.strengthened:
                s1 = self.added.union(self.strengthened)
                for idx, clause in enumerate(self.sentence):
                    if self.deleted[idx] or len(clause) > 2:
                        continue
                    for lit in clause:
                        if -lit in add_set:
                            s1.add(idx)
                            break
                self.added.clear()
                self.strengthened.clear()
                for idx in s1:
                    self.self_subsumed(self.sentence[idx])
                self.propagate_toplevel()
                flag = 0

            for idx in s0:
                if self.deleted[idx]:
                    continue
                self.subsumed(self.sentence[idx])

            while self.touched:
                s = set()
                for lit in self.touched:
                    if self.assigned[lit] or self.assigned[-lit]:
                        continue
                    if lit > 0:
                        s.add(lit)
                    else:
                        s.add(-lit)
                self.touched.clear()
                for lit in s:
                    self.maybe_eliminate(lit)
            if self.unsat:
                return None

        sentence = [list(clause) for idx, clause in enumerate(self.sentence) if not self.deleted[idx] and clause]
        b2s, s2b = {}, {}
        lit_dict = {lit: 0 for lit in range(1, self.num_vars + 1)}
        for clause in sentence:
            for lit in clause:
                lit_dict[abs(lit)] = 1
        lit_count = 0
        for lit in range(1, self.num_vars + 1):
            if lit_dict[lit]:
                lit_count += 1
                b2s[lit] = lit_count
                s2b[lit_count] = lit
            if not lit_dict[lit] and not (self.assigned[lit] or self.assigned[-lit]):
                self.assigned[lit] = 1
                self.assign_list.append(lit)

        assign_count = len(self.assign_list)
        for idx, clause in enumerate(sentence):
            sentence[idx] = self.translate(clause, b2s)
        t1 = time.time()
        print(f'Pretreatment takes time {t1-t0}')
        print(f'The clauses are reduced from {len(self.initial)} to {len(sentence)}')
        print(f'The variables are reduced from {self.num_vars} to {lit_count}')

        solver = Restarter(sentence, lit_count, self.decide_method, self.restart_method)
        extra_assign = solver.run()

        if not extra_assign:
            self.unsat = True
            return None
        extra_assign = self.translate(extra_assign, s2b)

        for lit in extra_assign:
            self.assign_list.append(lit)
            self.assigned[lit] = 1

        for idx in range(assign_count - 1, -1, -1):
            lit = self.assign_list[idx]
            if self.assigned[lit] == 1:
                continue
            self.assign_list[idx] *= -1
            for clause in self.assigned[lit]:
                if not self.satisfy(clause):
                    self.assign_list[idx] *= -1
                    break

            if self.assign_list[idx] == lit:
                self.assigned[lit] = 1
                self.assigned[-lit] = 0
            else:
                self.assigned[lit] = 0
                self.assigned[-lit] = 1

        assert self.tester() == 1
        return self.assign_list


class Restarter:
    def __init__(self, sentence, num_vars, decide_method='VSIDS', restart_method='Nothing'):
        self.sentence = sentence
        self.num_vars = num_vars

        self.decide_method = decide_method
        self.restart_method = restart_method
        self.choice = None

        # *************************************
        # variable for restart
        # ************************************
        self.conflict_threshold = num_vars/2
        self.time_threshold = 10
        self.num_arms = 3
        self.emp_means = np.zeros(self.num_arms)
        self.num_pulls = np.ones(self.num_arms)
        self.name = ['VSIDS', 'CHB', 'LRB']

    def run(self):
        if self.restart_method == 'Nothing':
            solver = Solver(self.sentence, self.num_vars,
                            self.decide_method, self.restart_method, self.conflict_threshold)
            flag, num_decisions, sentence_tmp, res = solver.run()
            return res
        elif self.restart_method == 'UCB':
            for t in range(self.time_threshold):
                if t < self.num_arms:
                    arm = t
                    solver = Solver(self.sentence, self.num_vars,
                                    self.name[arm], self.restart_method, self.conflict_threshold)
                    flag, num_decisions, sentence_tmp, res = solver.run()
                    if flag:
                        return res
                    else:
                        self.sentence = sentence_tmp
                        self.emp_means[arm] = math.log(num_decisions, 2)/len(res)
                    continue

                self.choice = np.zeros(self.num_arms)
                for i in range(self.num_arms):
                    self.choice[i] = self.emp_means[i] + math.sqrt(4*math.log(t)/self.num_pulls[i])
                arm = np.argmax(self.choice)
                solver = Solver(self.sentence, self.num_vars,
                                self.name[arm], self.restart_method, self.conflict_threshold)
                flag, num_decisions, sentence_tmp, res = solver.run()
                if flag:
                    return res
                else:
                    self.sentence = sentence_tmp
                    reward_tmp = math.log(num_decisions, 2)/len(res)
                    self.num_pulls[arm] += 1
                    self.emp_means[arm] = (self.emp_means[arm]*(self.num_pulls[arm]-1)+reward_tmp)/(self.num_pulls[arm])
            solver = Solver(self.sentence, self.num_vars,
                            self.decide_method, 'Nothing', self.conflict_threshold)
            flag, num_decisions, sentence_tmp, res = solver.run()
            return res
        elif self.restart_method == 'EXP3':
            weight = np.ones(self.num_arms)
            possible = np.ones(self.num_arms)
            g_value = 2*self.time_threshold/3
            gamma = min(1, int(math.sqrt((self.num_arms*math.log(self.num_arms))/((math.e-1)*g_value))))
            for t in range(self.time_threshold):
                for i in range(self.num_arms):
                    possible[i] = (1-gamma)*weight[i]/np.sum(weight) + gamma/self.num_arms
                ra = random.uniform(0, 1)
                curr_sum = 0
                curr_arm = 0
                for i in range(self.num_arms):
                    curr_sum += possible[i]
                    if ra <= curr_sum:
                        curr_arm = i
                        break
                solver = Solver(self.sentence, self.num_vars,
                                self.name[curr_arm], self.restart_method, self.conflict_threshold)
                flag, num_decisions, sentence_tmp, res = solver.run()
                if flag:
                    return res
                else:
                    self.sentence = sentence_tmp
                    reward_tmp = math.log(num_decisions, 2)/len(res)
                    weight[curr_arm] = weight[curr_arm] * math.exp(gamma*reward_tmp/self.num_arms)
            
            solver = Solver(self.sentence, self.num_vars,
                            self.decide_method, 'Nothing', self.conflict_threshold)
            flag, num_decisions, sentence_tmp, res = solver.run()
            return res


class Solver:
    def __init__(self, sentence, num_vars, method='LRB', restart_method='Nothing', conflict_threshold=1000):
        self.length = len(sentence)
        self.sentence = sentence
        self.num_vars = num_vars

        self.restart_method = restart_method
        self.conflict_threshold = conflict_threshold
        self.num_decisions = 0

        self.assign_map = {idx: 0 for idx in range(-num_vars, num_vars + 1)}
        self.ante_reason = {idx: [] for idx in range(-num_vars, num_vars + 1)}
        self.assignments, self.divide = [], []

        self.method = method
        self.uip = []
        self.num_conflicts = 0
        self.learn_counter = 0

        self.recorder = Recorder(sentence, num_vars)
        if method == 'VSIDS':
            self.decider = Vsids(self)
        elif method == 'CHB':
            self.decider = CHB(self)
        elif method == 'LRB':
            self.decider = LRB(self)

        #self.run()

    def bcp(self, new_assign=0, tag_clause=False):
        if tag_clause:
            new_assign = self.sentence[-1][-1]
            self.assignments.append(new_assign)
            self.assign_map[new_assign] = 1
            self.ante_reason[new_assign] = self.sentence[-1]
        # detect_area 中存的是clause的索引，该索引与sentence中的保持一致
        detect_area = self.recorder.l2c[-new_assign]
        extend = deque()
        self.recorder.detect(detect_area, self.assign_map, extend)

        while extend:
            (idx, new_assign) = extend.popleft()
            while extend and self.assign_map[new_assign]:
                (idx, new_assign) = extend.popleft()
            if not new_assign:  # 发现矛盾，返回矛盾
                return self.sentence[idx]
            if not extend:  # 没有要广播的变量
                return None
            self.assignments.append(new_assign)
            self.assign_map[new_assign] = 1
            self.ante_reason[new_assign] = self.sentence[idx]

            detect_area = self.recorder.l2c[-new_assign]
            self.recorder.detect(detect_area, self.assign_map, extend)
        return None

    def analyse(self, conflict):
        new_clause = set(conflict)
        conflict_level = len(self.divide)
        if not conflict_level:
            return -1, []

        idx_dict = dict()
        for idx, lit in enumerate(self.assignments):
            idx_dict[lit] = idx

        cur_level = set(self.assignments[self.divide[-1]:])  # 这一层的变量
        while True:
            related_lit = [-lit for lit in new_clause if -lit in cur_level]

            if len(related_lit) <= 1:  # 不用做resolve了
                break

            related_lit = sorted(related_lit, key=lambda literal: idx_dict[literal])
            lit = related_lit[-1]  # 最后被赋值的那个变量
            self.uip.append(lit)
            reason = self.ante_reason[lit]
            # 短短三行，实现了resolve函数，妙甚
            new_clause = new_clause.union(set(reason))
            new_clause.remove(lit)
            new_clause.remove(-lit)

        new_clause = sorted(list(new_clause), key=lambda literal: idx_dict[-literal])

        if len(new_clause) == 1:
            back_level = 0
        else:
            last2idx = idx_dict[-new_clause[-2]]
            back_level = bisect(self.divide, last2idx)

        return back_level, new_clause

    def backtrack(self, back_level):
        tail = self.divide[back_level]
        for i in range(tail, len(self.assignments)):
            self.assign_map[self.assignments[i]] = 0
            self.ante_reason[self.assignments[i]] = []

        if type(self.decider) == LRB:
            self.decider.back_work(self.assignments[tail:], self.learn_counter)

        self.assignments = self.assignments[:tail]
        self.divide = self.divide[:back_level]

    def run(self):
        if self.bcp():
            print(f'with {self.num_conflicts} conflict occurred')
            return True, self.num_decisions, self.sentence, None

        with tqdm(total=self.num_vars) as pbar:
            assign_pre = 0
            while len(self.assignments) < self.num_vars:
                pbar.update(len(self.assignments) - assign_pre)
                assign_pre = len(self.assignments)
                new_assign = self.decider.decide()
                self.divide.append(len(self.assignments))
                self.assignments.append(new_assign)
                self.num_decisions += 1
                self.assign_map[new_assign] = 1
                self.ante_reason[new_assign] = []
                conflict = self.bcp(new_assign)

                if type(self.decider) == CHB:
                    self.decider.record(new_assign, conflict)
                elif type(self.decider) == LRB:
                    self.decider.record(self.assignments[self.divide[-1]:],
                                        self.learn_counter)

                while conflict:
                    back_level, new_clause = self.analyse(conflict)

                    self.num_conflicts += 1
                    if self.num_conflicts > self.conflict_threshold:
                        if self.restart_method != 'Nothing':
                            return False, self.num_decisions, self.sentence, self.assignments

                    if type(self.decider) == Vsids:
                        self.decider.update(new_clause)
                    elif type(self.decider) == CHB:
                        self.decider.update(self.assignments[self.divide[back_level]:],
                                            self.num_conflicts, self.uip)
                    else:
                        self.learn_counter += 1
                        self.decider.update(conflict, new_clause, self.uip)

                    self.recorder.update(len(self.sentence), new_clause)
                    self.sentence.append(new_clause)
                    if back_level < 0:
                        print(f'with {self.num_conflicts} conflict occurred')
                        return True, self.num_decisions, self.sentence, None
                    self.backtrack(back_level)
                    conflict = self.bcp(0, True)
                if type(self.decider) == CHB:
                    self.decider.conclusion(self.num_conflicts)
                elif type(self.decider) == LRB:
                    self.decider.conclusion()
        print(f'with {self.num_conflicts} conflict occurred')
        return True, self.num_decisions, self.sentence, self.assignments


class Recorder:
    def __init__(self, sentence, num_vars):
        self.sentence = sentence
        self.c2l = {idx: sentence[idx][:2] for idx in range(0, len(sentence))}
        self.l2c = {idx: [] for idx in range(-num_vars, num_vars + 1)}

        for idx, clause in self.c2l.items():
            if len(clause) == 1:
                clause.append(None)

        for idx, claus in enumerate(sentence):
            self.l2c[0].append(idx)
            for lit in claus:
                self.l2c[lit].append(idx)

    @staticmethod
    def replace(clause, other, assign_map):
        '''
        从一个clause中找出还可以放到c2l中的变元
        :param clause: 一般取一个sentence中的句子
        :param other: c2l两个变元中的另一个
        :param assign_map:
        :return:
        '''
        for lit in clause:
            if not assign_map[-lit] and lit != other:
                return lit
        return None

    def update(self, idx, clause):
        for lit in clause:
            self.l2c[lit].append(idx)
        self.c2l[idx] = clause.copy()
        if len(self.c2l[idx]) == 1:
            self.c2l[idx].append(None)

    def detect(self, area, assign_map, extend):
        for idx in area:
            self.c2l[idx][0] = self.replace(self.sentence[idx], self.c2l[idx][1], assign_map)
            self.c2l[idx][1] = self.replace(self.sentence[idx], self.c2l[idx][0], assign_map)
            if not self.c2l[idx][0] and not self.c2l[idx][1]:  # 无法找到，此时矛盾产生
                extend.clear()
                extend.append((idx, 0))
                break
            if (self.c2l[idx][0] and assign_map[self.c2l[idx][0]]) \
                    or (self.c2l[idx][1] and assign_map[self.c2l[idx][1]]):
                continue
            if not self.c2l[idx][0]:
                extend.append((idx, self.c2l[idx][1]))
            if not self.c2l[idx][1]:
                extend.append((idx, self.c2l[idx][0]))


class Decider:
    def __init__(self, solver: Solver):
        self.assignments = solver.assignments
        self.sentence = solver.sentence
        self.num_vars = solver.num_vars
        self.assign_map = solver.assign_map
        self.scores = {idx: 0.0 for idx in range(-self.num_vars, self.num_vars + 1)}
        self.initialize()

    def initialize(self):
        pass

    def decide(self):
        pass


class Vsids(Decider):
    def __init__(self, solver: Solver):
        super().__init__(solver)

    def initialize(self):
        for clause in self.sentence:
            for lit in clause:
                self.scores[lit] += 1
        self.scores = dict(sorted(self.scores.items(), key=lambda kv: kv[1], reverse=True))

    def decide(self):
        for lit in self.scores.keys():
            if not self.assign_map[lit] and not self.assign_map[-lit]:
                return lit
        return None

    def update(self, clause, decay=0.95):
        for lit in self.scores:
            self.scores[lit] *= decay
        for lit in clause:
            self.scores[lit] += 1

        self.scores = dict(sorted(self.scores.items(), key=lambda kv: kv[1], reverse=True))


class CHB(Decider):
    def __init__(self, solver: Solver):
        super().__init__(solver)
        self.multiplier = None
        self.alpha = 0.4
        self.assigned_prev = None
        self.plays = set()
        self.last_conflict = {lit: 0 for lit in range(-self.num_vars, self.num_vars + 1)}

    def record(self, new_assign, conflict):
        self.plays = {new_assign}
        self.assigned_prev = set(self.assignments)
        self.multiplier = 1.0 if conflict is not None else 0.9

    def update(self, area, num_conflicts, uip):
        if self.alpha > 0.06:
            self.alpha -= 1e-6
        for lit in area:
            self.last_conflict[lit] = num_conflicts
        self.plays = set(uip)

    def decide(self):
        for lit in self.scores.keys():
            if not self.assign_map[lit] and not self.assign_map[-lit]:
                return lit
        return None

    def conclusion(self, num_conflicts):
        for v in self.plays.union(self.assigned_prev.difference(set(self.assignments))):
            reward = self.multiplier / (num_conflicts - self.last_conflict[v] + 1)
            self.scores[v] = (1 - self.alpha) * self.scores[v] + self.alpha * reward
        self.scores = dict(sorted(self.scores.items(), key=lambda kv: kv[1], reverse=True))


class LRB(Decider):
    def __init__(self, solver: Solver):
        super().__init__(solver)
        self.alpha = 0.4
        self.assigned = {lit: 0 for lit in range(-self.num_vars, self.num_vars + 1)}
        self.participated = {lit: 0 for lit in range(-self.num_vars, self.num_vars + 1)}
        self.reasoned = {lit: 0 for lit in range(-self.num_vars, self.num_vars + 1)}

    def decide(self):
        for lit in self.scores.keys():
            if not self.assign_map[lit] and not self.assign_map[-lit]:
                return lit
        return None

    def record(self, area, learn_count):
        for lit in area:
            self.assigned[lit] = learn_count
            self.participated[lit] = 0
            self.reasoned[lit] = 0

    def update(self, conflict, new_clause, uip):
        if self.alpha > 0.06:
            self.alpha -= 1e-6
        for lit in set(new_clause).union(set(conflict)):
            self.participated[lit] += 1
        for lit in set(uip).difference(set(new_clause)):
            self.reasoned[lit] += 1

    def back_work(self, area, learn_count):
        for lit in area:
            interval = learn_count - self.assigned[lit]
            if interval > 0:
                r = self.participated[lit] / interval
                rsr = self.reasoned[lit] / interval
                self.scores[lit] = (1 - self.alpha) * self.scores[lit] + self.alpha * (r + rsr)

    def conclusion(self, decay=0.95):
        for lit in self.scores:
            if not self.assign_map[lit]:
                self.scores[lit] = self.scores[lit] * decay
        self.scores = dict(sorted(self.scores.items(), key=lambda kv: kv[1], reverse=True))
