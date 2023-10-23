import openai
import pyttsx3
import speech_recognition as sr
import webbrowser
import yaml
import os
import sqlite3
import importlib
import random
import pytest
from utils import failure_test
from csp import *
import random
import pytest
from knowledge import *
from utils import expr
import random
from keras.datasets import imdb
from deep_learning4e import *
from learning4e import DataSet, grade_learner, err_ratio

from agents import (ReflexVacuumAgent, ModelBasedVacuumAgent, TrivialVacuumEnvironment, compare_agents,
                    RandomVacuumAgent, TableDrivenVacuumAgent, TableDrivenAgentProgram, RandomAgentProgram,
                    SimpleReflexAgentProgram, ModelBasedReflexAgentProgram, Wall, Gold, Explorer, Thing, Bump, Glitter,
                    WumpusEnvironment, Pit, VacuumEnvironment, Dirt, Direction, Agent)

# Baca konfigurasi API dari file YAML
with open("config/api_config.yaml", "r") as yaml_file:
    api_data = yaml.safe_load(yaml_file)
openai.api_key = api_data["openai"]["api_key"]
completion = openai.Completion()
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')

# Inisialisasi koneksi ke database SQLite
db_connection = sqlite3.connect("data/database/user_data.db")
db_cursor = db_connection.cursor()

def import_src_modules(src_dir):
    for root, dirs, files in os.walk(src_dir):
        # Loop through all Python files in the directory
        for file in files:
            if file.endswith('.py') and file != "__init__.py":
                # Construct the full module path by replacing slashes with dots
                module_path = os.path.relpath(os.path.join(root, file), src_dir)
                module_path = os.path.splitext(module_path)[0].replace(os.path.sep, '.')
                # Import the module dynamically
                importlib.import_module(module_path)

def initialize_database():
    # Buat tabel jika belum ada
    db_cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_data (
        id INTEGER PRIMARY KEY,
        name TEXT,
        query TEXT
    )''')
    db_connection.commit()

def add_user_data(name, query):
    # Tambahkan data pengguna ke database
    db_cursor.execute('INSERT INTO user_data (name, query) VALUES (?, ?)', (name, query))

def Reply(question):
    prompt = f'Chando: {question}\n DANI: '
    response = completion.create(prompt=prompt, engine="text-davinci-002", stop=['\Chando'], max_tokens=200)
    answer = response.choices[0].text.strip()
    return answer
def test_move_forward():
    d = Direction("up")
    l1 = d.move_forward((0, 0))
    assert l1 == (0, -1)

    d = Direction(Direction.R)
    l1 = d.move_forward((0, 0))
    assert l1 == (1, 0)

    d = Direction(Direction.D)
    l1 = d.move_forward((0, 0))
    assert l1 == (0, 1)

    d = Direction("left")
    l1 = d.move_forward((0, 0))
    assert l1 == (-1, 0)

    l2 = d.move_forward((1, 0))
    assert l2 == (0, 0)


def test_add():
    d = Direction(Direction.U)
    l1 = d + "right"
    l2 = d + "left"
    assert l1.direction == Direction.R
    assert l2.direction == Direction.L

    d = Direction("right")
    l1 = d.__add__(Direction.L)
    l2 = d.__add__(Direction.R)
    assert l1.direction == "up"
    assert l2.direction == "down"

    d = Direction("down")
    l1 = d.__add__("right")
    l2 = d.__add__("left")
    assert l1.direction == Direction.L
    assert l2.direction == Direction.R

    d = Direction(Direction.L)
    l1 = d + Direction.R
    l2 = d + Direction.L
    assert l1.direction == Direction.U
    assert l2.direction == Direction.D


def test_RandomAgentProgram():
    # create a list of all the actions a Vacuum cleaner can perform
    list = ['Right', 'Left', 'Suck', 'NoOp']
    # create a program and then an object of the RandomAgentProgram
    program = RandomAgentProgram(list)

    agent = Agent(program)
    # create an object of TrivialVacuumEnvironment
    environment = TrivialVacuumEnvironment()
    # add agent to the environment
    environment.add_thing(agent)
    # run the environment
    environment.run()
    # check final status of the environment
    assert environment.status == {(1, 0): 'Clean', (0, 0): 'Clean'}


def test_RandomVacuumAgent():
    # create an object of the RandomVacuumAgent
    agent = RandomVacuumAgent()
    # create an object of TrivialVacuumEnvironment
    environment = TrivialVacuumEnvironment()
    # add agent to the environment
    environment.add_thing(agent)
    # run the environment
    environment.run()
    # check final status of the environment
    assert environment.status == {(1, 0): 'Clean', (0, 0): 'Clean'}


def test_TableDrivenAgent():
    random.seed(10)
    loc_A, loc_B = (0, 0), (1, 0)
    # table defining all the possible states of the agent
    table = {((loc_A, 'Clean'),): 'Right',
             ((loc_A, 'Dirty'),): 'Suck',
             ((loc_B, 'Clean'),): 'Left',
             ((loc_B, 'Dirty'),): 'Suck',
             ((loc_A, 'Dirty'), (loc_A, 'Clean')): 'Right',
             ((loc_A, 'Clean'), (loc_B, 'Dirty')): 'Suck',
             ((loc_B, 'Clean'), (loc_A, 'Dirty')): 'Suck',
             ((loc_B, 'Dirty'), (loc_B, 'Clean')): 'Left',
             ((loc_A, 'Dirty'), (loc_A, 'Clean'), (loc_B, 'Dirty')): 'Suck',
             ((loc_B, 'Dirty'), (loc_B, 'Clean'), (loc_A, 'Dirty')): 'Suck'}

    # create an program and then an object of the TableDrivenAgent
    program = TableDrivenAgentProgram(table)
    agent = Agent(program)
    # create an object of TrivialVacuumEnvironment
    environment = TrivialVacuumEnvironment()
    # initializing some environment status
    environment.status = {loc_A: 'Dirty', loc_B: 'Dirty'}
    # add agent to the environment
    environment.add_thing(agent)

    # run the environment by single step everytime to check how environment evolves using TableDrivenAgentProgram
    environment.run(steps=1)
    assert environment.status == {(1, 0): 'Clean', (0, 0): 'Dirty'}

    environment.run(steps=1)
    assert environment.status == {(1, 0): 'Clean', (0, 0): 'Dirty'}

    environment.run(steps=1)
    assert environment.status == {(1, 0): 'Clean', (0, 0): 'Clean'}


def test_ReflexVacuumAgent():
    # create an object of the ReflexVacuumAgent
    agent = ReflexVacuumAgent()
    # create an object of TrivialVacuumEnvironment
    environment = TrivialVacuumEnvironment()
    # add agent to the environment
    environment.add_thing(agent)
    # run the environment
    environment.run()
    # check final status of the environment
    assert environment.status == {(1, 0): 'Clean', (0, 0): 'Clean'}


def test_SimpleReflexAgentProgram():
    class Rule:

        def __init__(self, state, action):
            self.__state = state
            self.action = action

        def matches(self, state):
            return self.__state == state

    loc_A = (0, 0)
    loc_B = (1, 0)

    # create rules for a two state Vacuum Environment
    rules = [Rule((loc_A, "Dirty"), "Suck"), Rule((loc_A, "Clean"), "Right"),
             Rule((loc_B, "Dirty"), "Suck"), Rule((loc_B, "Clean"), "Left")]

    def interpret_input(state):
        return state

    # create a program and then an object of the SimpleReflexAgentProgram
    program = SimpleReflexAgentProgram(rules, interpret_input)
    agent = Agent(program)
    # create an object of TrivialVacuumEnvironment
    environment = TrivialVacuumEnvironment()
    # add agent to the environment
    environment.add_thing(agent)
    # run the environment
    environment.run()
    # check final status of the environment
    assert environment.status == {(1, 0): 'Clean', (0, 0): 'Clean'}


def test_ModelBasedReflexAgentProgram():
    class Rule:

        def __init__(self, state, action):
            self.__state = state
            self.action = action

        def matches(self, state):
            return self.__state == state

    loc_A = (0, 0)
    loc_B = (1, 0)

    # create rules for a two-state Vacuum Environment
    rules = [Rule((loc_A, "Dirty"), "Suck"), Rule((loc_A, "Clean"), "Right"),
             Rule((loc_B, "Dirty"), "Suck"), Rule((loc_B, "Clean"), "Left")]

    def update_state(state, action, percept, model):
        return percept

    # create a program and then an object of the ModelBasedReflexAgentProgram class
    program = ModelBasedReflexAgentProgram(rules, update_state, None)
    agent = Agent(program)
    # create an object of TrivialVacuumEnvironment
    environment = TrivialVacuumEnvironment()
    # add agent to the environment
    environment.add_thing(agent)
    # run the environment
    environment.run()
    # check final status of the environment
    assert environment.status == {(1, 0): 'Clean', (0, 0): 'Clean'}


def test_ModelBasedVacuumAgent():
    # create an object of the ModelBasedVacuumAgent
    agent = ModelBasedVacuumAgent()
    # create an object of TrivialVacuumEnvironment
    environment = TrivialVacuumEnvironment()
    # add agent to the environment
    environment.add_thing(agent)
    # run the environment
    environment.run()
    # check final status of the environment
    assert environment.status == {(1, 0): 'Clean', (0, 0): 'Clean'}


def test_TableDrivenVacuumAgent():
    # create an object of the TableDrivenVacuumAgent
    agent = TableDrivenVacuumAgent()
    # create an object of the TrivialVacuumEnvironment
    environment = TrivialVacuumEnvironment()
    # add agent to the environment
    environment.add_thing(agent)
    # run the environment
    environment.run()
    # check final status of the environment
    assert environment.status == {(1, 0): 'Clean', (0, 0): 'Clean'}


def test_compare_agents():
    environment = TrivialVacuumEnvironment
    agents = [ModelBasedVacuumAgent, ReflexVacuumAgent]

    result = compare_agents(environment, agents)
    performance_ModelBasedVacuumAgent = result[0][1]
    performance_ReflexVacuumAgent = result[1][1]

    # The performance of ModelBasedVacuumAgent will be at least as good as that of
    # ReflexVacuumAgent, since ModelBasedVacuumAgent can identify when it has
    # reached the terminal state (both locations being clean) and will perform
    # NoOp leading to 0 performance change, whereas ReflexVacuumAgent cannot
    # identify the terminal state and thus will keep moving, leading to worse
    # performance compared to ModelBasedVacuumAgent.
    assert performance_ReflexVacuumAgent <= performance_ModelBasedVacuumAgent


def test_TableDrivenAgentProgram():
    table = {(('foo', 1),): 'action1',
             (('foo', 2),): 'action2',
             (('bar', 1),): 'action3',
             (('bar', 2),): 'action1',
             (('foo', 1), ('foo', 1),): 'action2',
             (('foo', 1), ('foo', 2),): 'action3'}
    agent_program = TableDrivenAgentProgram(table)
    assert agent_program(('foo', 1)) == 'action1'
    assert agent_program(('foo', 2)) == 'action3'
    assert agent_program(('invalid percept',)) is None

def test_Agent():
    def constant_prog(percept):
        return percept

    agent = Agent(constant_prog)
    result = agent.program(5)
    assert result == 5

def test_VacuumEnvironment():
    # initialize Vacuum Environment
    v = VacuumEnvironment(6, 6)
    # get an agent
    agent = ModelBasedVacuumAgent()
    agent.direction = Direction(Direction.R)
    v.add_thing(agent)
    v.add_thing(Dirt(), location=(2, 1))

    # check if things are added properly
    assert len([x for x in v.things if isinstance(x, Wall)]) == 20
    assert len([x for x in v.things if isinstance(x, Dirt)]) == 1

    # let the action begin!
    assert v.percept(agent) == ("Clean", "None")
    v.execute_action(agent, "Forward")
    assert v.percept(agent) == ("Dirty", "None")
    v.execute_action(agent, "TurnLeft")
    v.execute_action(agent, "Forward")
    assert v.percept(agent) == ("Dirty", "Bump")
    v.execute_action(agent, "Suck")
    assert v.percept(agent) == ("Clean", "None")
    old_performance = agent.performance
    v.execute_action(agent, "NoOp")
    assert old_performance == agent.performance


def test_WumpusEnvironment():
    def constant_prog(percept):
        return percept

    # initialize Wumpus Environment
    w = WumpusEnvironment(constant_prog)

    # check if things are added properly
    assert len([x for x in w.things if isinstance(x, Wall)]) == 20
    assert any(map(lambda x: isinstance(x, Gold), w.things))
    assert any(map(lambda x: isinstance(x, Explorer), w.things))
    assert not any(map(lambda x: not isinstance(x, Thing), w.things))

    # check that gold and wumpus are not present on (1,1)
    assert not any(map(lambda x: isinstance(x, Gold) or isinstance(x, WumpusEnvironment), w.list_things_at((1, 1))))

    # check if w.get_world() segments objects correctly
    assert len(w.get_world()) == 6
    for row in w.get_world():
        assert len(row) == 6

    # start the game!
    agent = [x for x in w.things if isinstance(x, Explorer)][0]
    gold = [x for x in w.things if isinstance(x, Gold)][0]
    pit = [x for x in w.things if isinstance(x, Pit)][0]

    assert not w.is_done()

    # check Walls
    agent.location = (1, 2)
    percepts = w.percept(agent)
    assert len(percepts) == 5
    assert any(map(lambda x: isinstance(x, Bump), percepts[0]))

    # check Gold
    agent.location = gold.location
    percepts = w.percept(agent)
    assert any(map(lambda x: isinstance(x, Glitter), percepts[4]))
    agent.location = (gold.location[0], gold.location[1] + 1)
    percepts = w.percept(agent)
    assert not any(map(lambda x: isinstance(x, Glitter), percepts[4]))

    # check agent death
    agent.location = pit.location
    assert w.in_danger(agent)
    assert not agent.alive
    assert agent.killed_by == Pit.__name__
    assert agent.performance == -1000

    assert w.is_done()


def test_WumpusEnvironmentActions():
    random.seed(9)
    def constant_prog(percept):
        return percept

    # initialize Wumpus Environment
    w = WumpusEnvironment(constant_prog)

    agent = [x for x in w.things if isinstance(x, Explorer)][0]
    gold = [x for x in w.things if isinstance(x, Gold)][0]
    pit = [x for x in w.things if isinstance(x, Pit)][0]

    agent.location = (1, 1)
    assert agent.direction.direction == "right"
    w.execute_action(agent, 'TurnRight')
    assert agent.direction.direction == "down"
    w.execute_action(agent, 'TurnLeft')
    assert agent.direction.direction == "right"
    w.execute_action(agent, 'Forward')
    assert agent.location == (2, 1)

    agent.location = gold.location
    w.execute_action(agent, 'Grab')
    assert agent.holding == [gold]

    agent.location = (1, 1)
    w.execute_action(agent, 'Climb')
    assert not any(map(lambda x: isinstance(x, Explorer), w.things))

    assert w.is_done()
def test_csp_assign():
    var = 10
    val = 5
    assignment = {}
    australia_csp.assign(var, val, assignment)

    assert australia_csp.nassigns == 1
    assert assignment[var] == val


def test_csp_unassign():
    var = 10
    assignment = {var: 5}
    australia_csp.unassign(var, assignment)

    assert var not in assignment


def test_csp_nconflicts():
    map_coloring_test = MapColoringCSP(list('RGB'), 'A: B C; B: C; C: ')
    assignment = {'A': 'R', 'B': 'G'}
    var = 'C'
    val = 'R'
    assert map_coloring_test.nconflicts(var, val, assignment) == 1

    val = 'B'
    assert map_coloring_test.nconflicts(var, val, assignment) == 0


def test_csp_actions():
    map_coloring_test = MapColoringCSP(list('123'), 'A: B C; B: C; C: ')

    state = {'A': '1', 'B': '2', 'C': '3'}
    assert map_coloring_test.actions(state) == []

    state = {'A': '1', 'B': '3'}
    assert map_coloring_test.actions(state) == [('C', '2')]

    state = {'A': '1', 'C': '2'}
    assert map_coloring_test.actions(state) == [('B', '3')]

    state = (('A', '1'), ('B', '3'))
    assert map_coloring_test.actions(state) == [('C', '2')]

    state = {'A': '1'}
    assert (map_coloring_test.actions(state) == [('C', '2'), ('C', '3')] or
            map_coloring_test.actions(state) == [('B', '2'), ('B', '3')])


def test_csp_result():
    map_coloring_test = MapColoringCSP(list('123'), 'A: B C; B: C; C: ')

    state = (('A', '1'), ('B', '3'))
    action = ('C', '2')

    assert map_coloring_test.result(state, action) == (('A', '1'), ('B', '3'), ('C', '2'))


def test_csp_goal_test():
    map_coloring_test = MapColoringCSP(list('123'), 'A: B C; B: C; C: ')
    state = (('A', '1'), ('B', '3'), ('C', '2'))
    assert map_coloring_test.goal_test(state)

    state = (('A', '1'), ('C', '2'))
    assert not map_coloring_test.goal_test(state)


def test_csp_support_pruning():
    map_coloring_test = MapColoringCSP(list('123'), 'A: B C; B: C; C: ')
    map_coloring_test.support_pruning()
    assert map_coloring_test.curr_domains == {'A': ['1', '2', '3'], 'B': ['1', '2', '3'], 'C': ['1', '2', '3']}


def test_csp_suppose():
    map_coloring_test = MapColoringCSP(list('123'), 'A: B C; B: C; C: ')
    var = 'A'
    value = '1'

    removals = map_coloring_test.suppose(var, value)

    assert removals == [('A', '2'), ('A', '3')]
    assert map_coloring_test.curr_domains == {'A': ['1'], 'B': ['1', '2', '3'], 'C': ['1', '2', '3']}


def test_csp_prune():
    map_coloring_test = MapColoringCSP(list('123'), 'A: B C; B: C; C: ')
    removals = None
    var = 'A'
    value = '3'

    map_coloring_test.support_pruning()
    map_coloring_test.prune(var, value, removals)
    assert map_coloring_test.curr_domains == {'A': ['1', '2'], 'B': ['1', '2', '3'], 'C': ['1', '2', '3']}
    assert removals is None

    map_coloring_test = MapColoringCSP(list('123'), 'A: B C; B: C; C: ')
    removals = [('A', '2')]
    map_coloring_test.support_pruning()
    map_coloring_test.prune(var, value, removals)
    assert map_coloring_test.curr_domains == {'A': ['1', '2'], 'B': ['1', '2', '3'], 'C': ['1', '2', '3']}
    assert removals == [('A', '2'), ('A', '3')]


def test_csp_choices():
    map_coloring_test = MapColoringCSP(list('123'), 'A: B C; B: C; C: ')
    var = 'A'
    assert map_coloring_test.choices(var) == ['1', '2', '3']

    map_coloring_test.support_pruning()
    removals = None
    value = '3'
    map_coloring_test.prune(var, value, removals)
    assert map_coloring_test.choices(var) == ['1', '2']


def test_csp_infer_assignment():
    map_coloring_test = MapColoringCSP(list('123'), 'A: B C; B: C; C: ')
    assert map_coloring_test.infer_assignment() == {}

    var = 'A'
    value = '3'
    map_coloring_test.prune(var, value, None)
    value = '1'
    map_coloring_test.prune(var, value, None)

    assert map_coloring_test.infer_assignment() == {'A': '2'}


def test_csp_restore():
    map_coloring_test = MapColoringCSP(list('123'), 'A: B C; B: C; C: ')
    map_coloring_test.curr_domains = {'A': ['2', '3'], 'B': ['1'], 'C': ['2', '3']}
    removals = [('A', '1'), ('B', '2'), ('B', '3')]

    map_coloring_test.restore(removals)

    assert map_coloring_test.curr_domains == {'A': ['2', '3', '1'], 'B': ['1', '2', '3'], 'C': ['2', '3']}


def test_csp_conflicted_vars():
    map_coloring_test = MapColoringCSP(list('123'), 'A: B C; B: C; C: ')

    current = {}
    var = 'A'
    val = '1'
    map_coloring_test.assign(var, val, current)

    var = 'B'
    val = '3'
    map_coloring_test.assign(var, val, current)

    var = 'C'
    val = '3'
    map_coloring_test.assign(var, val, current)

    conflicted_vars = map_coloring_test.conflicted_vars(current)

    assert (conflicted_vars == ['B', 'C'] or conflicted_vars == ['C', 'B'])


def test_revise():
    neighbors = parse_neighbors('A: B; B: ')
    domains = {'A': [0], 'B': [4]}
    constraints = lambda X, x, Y, y: x % 2 == 0 and (x + y) == 4

    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)
    csp.support_pruning()
    Xi = 'A'
    Xj = 'B'
    removals = []

    consistency, _ = revise(csp, Xi, Xj, removals)
    assert not consistency
    assert len(removals) == 0

    domains = {'A': [0, 1, 2, 3, 4], 'B': [0, 1, 2, 3, 4]}
    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)
    csp.support_pruning()

    assert revise(csp, Xi, Xj, removals)
    assert removals == [('A', 1), ('A', 3)]


def test_AC3():
    neighbors = parse_neighbors('A: B; B: ')
    domains = {'A': [0, 1, 2, 3, 4], 'B': [0, 1, 2, 3, 4]}
    constraints = lambda X, x, Y, y: x % 2 == 0 and x + y == 4 and y % 2 != 0
    removals = []

    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)

    consistency, _ = AC3(csp, removals=removals)
    assert not consistency

    constraints = lambda X, x, Y, y: x % 2 == 0 and x + y == 4
    removals = []
    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)

    assert AC3(csp, removals=removals)
    assert (removals == [('A', 1), ('A', 3), ('B', 1), ('B', 3)] or
            removals == [('B', 1), ('B', 3), ('A', 1), ('A', 3)])

    domains = {'A': [2, 4], 'B': [3, 5]}
    constraints = lambda X, x, Y, y: (X == 'A' and Y == 'B') or (X == 'B' and Y == 'A') and x > y
    removals = []
    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)

    assert AC3(csp, removals=removals)


def test_AC3b():
    neighbors = parse_neighbors('A: B; B: ')
    domains = {'A': [0, 1, 2, 3, 4], 'B': [0, 1, 2, 3, 4]}
    constraints = lambda X, x, Y, y: x % 2 == 0 and x + y == 4 and y % 2 != 0
    removals = []

    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)

    consistency, _ = AC3b(csp, removals=removals)
    assert not consistency

    constraints = lambda X, x, Y, y: x % 2 == 0 and x + y == 4
    removals = []
    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)

    assert AC3b(csp, removals=removals)
    assert (removals == [('A', 1), ('A', 3), ('B', 1), ('B', 3)] or
            removals == [('B', 1), ('B', 3), ('A', 1), ('A', 3)])

    domains = {'A': [2, 4], 'B': [3, 5]}
    constraints = lambda X, x, Y, y: (X == 'A' and Y == 'B') or (X == 'B' and Y == 'A') and x > y
    removals = []
    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)

    assert AC3b(csp, removals=removals)


def test_AC4():
    neighbors = parse_neighbors('A: B; B: ')
    domains = {'A': [0, 1, 2, 3, 4], 'B': [0, 1, 2, 3, 4]}
    constraints = lambda X, x, Y, y: x % 2 == 0 and x + y == 4 and y % 2 != 0
    removals = []

    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)

    consistency, _ = AC4(csp, removals=removals)
    assert not consistency

    constraints = lambda X, x, Y, y: x % 2 == 0 and x + y == 4
    removals = []
    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)

    assert AC4(csp, removals=removals)
    assert (removals == [('A', 1), ('A', 3), ('B', 1), ('B', 3)] or
            removals == [('B', 1), ('B', 3), ('A', 1), ('A', 3)])

    domains = {'A': [2, 4], 'B': [3, 5]}
    constraints = lambda X, x, Y, y: (X == 'A' and Y == 'B') or (X == 'B' and Y == 'A') and x > y
    removals = []
    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)

    assert AC4(csp, removals=removals)


def test_first_unassigned_variable():
    map_coloring_test = MapColoringCSP(list('123'), 'A: B C; B: C; C: ')
    assignment = {'A': '1', 'B': '2'}
    assert first_unassigned_variable(assignment, map_coloring_test) == 'C'

    assignment = {'B': '1'}
    assert (first_unassigned_variable(assignment, map_coloring_test) == 'A' or
            first_unassigned_variable(assignment, map_coloring_test) == 'C')


def test_num_legal_values():
    map_coloring_test = MapColoringCSP(list('123'), 'A: B C; B: C; C: ')
    map_coloring_test.support_pruning()
    var = 'A'
    assignment = {}

    assert num_legal_values(map_coloring_test, var, assignment) == 3

    map_coloring_test = MapColoringCSP(list('RGB'), 'A: B C; B: C; C: ')
    assignment = {'A': 'R', 'B': 'G'}
    var = 'C'

    assert num_legal_values(map_coloring_test, var, assignment) == 1


def test_mrv():
    neighbors = parse_neighbors('A: B; B: C; C: ')
    domains = {'A': [0, 1, 2, 3, 4], 'B': [4], 'C': [0, 1, 2, 3, 4]}
    constraints = lambda X, x, Y, y: x % 2 == 0 and x + y == 4
    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)
    assignment = {'A': 0}

    assert mrv(assignment, csp) == 'B'

    domains = {'A': [0, 1, 2, 3, 4], 'B': [0, 1, 2, 3, 4], 'C': [0, 1, 2, 3, 4]}
    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)

    assert (mrv(assignment, csp) == 'B' or
            mrv(assignment, csp) == 'C')

    domains = {'A': [0, 1, 2, 3, 4], 'B': [0, 1, 2, 3, 4, 5, 6], 'C': [0, 1, 2, 3, 4]}
    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)
    csp.support_pruning()

    assert mrv(assignment, csp) == 'C'


def test_unordered_domain_values():
    map_coloring_test = MapColoringCSP(list('123'), 'A: B C; B: C; C: ')
    assignment = None
    assert unordered_domain_values('A', assignment, map_coloring_test) == ['1', '2', '3']


def test_lcv():
    neighbors = parse_neighbors('A: B; B: C; C: ')
    domains = {'A': [0, 1, 2, 3, 4], 'B': [0, 1, 2, 3, 4, 5], 'C': [0, 1, 2, 3, 4]}
    constraints = lambda X, x, Y, y: x % 2 == 0 and (x + y) == 4
    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)
    assignment = {'A': 0}

    var = 'B'

    assert lcv(var, assignment, csp) == [4, 0, 1, 2, 3, 5]
    assignment = {'A': 1, 'C': 3}

    constraints = lambda X, x, Y, y: (x + y) % 2 == 0 and (x + y) < 5
    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)

    assert lcv(var, assignment, csp) == [1, 3, 0, 2, 4, 5]


def test_forward_checking():
    neighbors = parse_neighbors('A: B; B: C; C: ')
    domains = {'A': [0, 1, 2, 3, 4], 'B': [0, 1, 2, 3, 4, 5], 'C': [0, 1, 2, 3, 4]}
    constraints = lambda X, x, Y, y: (x + y) % 2 == 0 and (x + y) < 8
    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)

    csp.support_pruning()
    A_curr_domains = csp.curr_domains['A']
    C_curr_domains = csp.curr_domains['C']

    var = 'B'
    value = 3
    assignment = {'A': 1, 'C': '3'}
    assert forward_checking(csp, var, value, assignment, None)
    assert csp.curr_domains['A'] == A_curr_domains
    assert csp.curr_domains['C'] == C_curr_domains

    assignment = {'C': 3}

    assert forward_checking(csp, var, value, assignment, None)
    assert csp.curr_domains['A'] == [1, 3]

    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)
    csp.support_pruning()

    assignment = {}
    assert forward_checking(csp, var, value, assignment, None)
    assert csp.curr_domains['A'] == [1, 3]
    assert csp.curr_domains['C'] == [1, 3]

    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)
    csp.support_pruning()

    value = 7
    assignment = {}
    assert not forward_checking(csp, var, value, assignment, None)
    assert (csp.curr_domains['A'] == [] or csp.curr_domains['C'] == [])


def test_backtracking_search():
    assert backtracking_search(australia_csp)
    assert backtracking_search(australia_csp, select_unassigned_variable=mrv)
    assert backtracking_search(australia_csp, order_domain_values=lcv)
    assert backtracking_search(australia_csp, select_unassigned_variable=mrv, order_domain_values=lcv)
    assert backtracking_search(australia_csp, inference=forward_checking)
    assert backtracking_search(australia_csp, inference=mac)
    assert backtracking_search(usa_csp, select_unassigned_variable=mrv, order_domain_values=lcv, inference=mac)


def test_min_conflicts():
    assert min_conflicts(australia_csp)
    assert min_conflicts(france_csp)

    tests = [(usa_csp, None)] * 3
    assert failure_test(min_conflicts, tests) >= 1 / 3

    australia_impossible = MapColoringCSP(list('RG'), 'SA: WA NT Q NSW V; NT: WA Q; NSW: Q V; T: ')
    assert min_conflicts(australia_impossible, 1000) is None
    assert min_conflicts(NQueensCSP(2), 1000) is None
    assert min_conflicts(NQueensCSP(3), 1000) is None


def test_nqueens_csp():
    csp = NQueensCSP(8)

    assignment = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    csp.assign(5, 5, assignment)
    assert len(assignment) == 6
    csp.assign(6, 6, assignment)
    assert len(assignment) == 7
    csp.assign(7, 7, assignment)
    assert len(assignment) == 8
    assert assignment[5] == 5
    assert assignment[6] == 6
    assert assignment[7] == 7
    assert csp.nconflicts(3, 2, assignment) == 0
    assert csp.nconflicts(3, 3, assignment) == 0
    assert csp.nconflicts(1, 5, assignment) == 1
    assert csp.nconflicts(7, 5, assignment) == 2
    csp.unassign(1, assignment)
    csp.unassign(2, assignment)
    csp.unassign(3, assignment)
    assert 1 not in assignment
    assert 2 not in assignment
    assert 3 not in assignment

    assignment = {0: 0, 1: 1, 2: 4, 3: 1, 4: 6}
    csp.assign(5, 7, assignment)
    assert len(assignment) == 6
    csp.assign(6, 6, assignment)
    assert len(assignment) == 7
    csp.assign(7, 2, assignment)
    assert len(assignment) == 8
    assert assignment[5] == 7
    assert assignment[6] == 6
    assert assignment[7] == 2
    assignment = {0: 0, 1: 1, 2: 4, 3: 1, 4: 6, 5: 7, 6: 6, 7: 2}
    assert csp.nconflicts(7, 7, assignment) == 4
    assert csp.nconflicts(3, 4, assignment) == 0
    assert csp.nconflicts(2, 6, assignment) == 2
    assert csp.nconflicts(5, 5, assignment) == 3
    csp.unassign(4, assignment)
    csp.unassign(5, assignment)
    csp.unassign(6, assignment)
    assert 4 not in assignment
    assert 5 not in assignment
    assert 6 not in assignment

    for n in range(5, 9):
        csp = NQueensCSP(n)
        solution = min_conflicts(csp)
        assert not solution or sorted(solution.values()) == list(range(n))


def test_universal_dict():
    d = UniversalDict(42)
    assert d['life'] == 42


def test_parse_neighbours():
    assert parse_neighbors('X: Y Z; Y: Z') == {'Y': ['X', 'Z'], 'X': ['Y', 'Z'], 'Z': ['X', 'Y']}


def test_topological_sort():
    root = 'NT'
    Sort, Parents = topological_sort(australia_csp, root)

    assert Sort == ['NT', 'SA', 'Q', 'NSW', 'V', 'WA']
    assert Parents['NT'] is None
    assert Parents['SA'] == 'NT'
    assert Parents['Q'] == 'SA'
    assert Parents['NSW'] == 'Q'
    assert Parents['V'] == 'NSW'
    assert Parents['WA'] == 'SA'


def test_tree_csp_solver():
    australia_small = MapColoringCSP(list('RB'), 'NT: WA Q; NSW: Q V')
    tcs = tree_csp_solver(australia_small)
    assert (tcs['NT'] == 'R' and tcs['WA'] == 'B' and tcs['Q'] == 'B' and tcs['NSW'] == 'R' and tcs['V'] == 'B') or \
           (tcs['NT'] == 'B' and tcs['WA'] == 'R' and tcs['Q'] == 'R' and tcs['NSW'] == 'B' and tcs['V'] == 'R')


def test_ac_solver():
    assert ac_solver(csp_crossword) == {'one_across': 'has',
                                        'one_down': 'hold',
                                        'two_down': 'syntax',
                                        'three_across': 'land',
                                        'four_across': 'ant'} or {'one_across': 'bus',
                                                                  'one_down': 'buys',
                                                                  'two_down': 'search',
                                                                  'three_across': 'year',
                                                                  'four_across': 'car'}
    assert ac_solver(two_two_four) == {'T': 7, 'F': 1, 'W': 6, 'O': 5, 'U': 3, 'R': 0, 'C1': 1, 'C2': 1, 'C3': 1} or \
           {'T': 9, 'F': 1, 'W': 2, 'O': 8, 'U': 5, 'R': 6, 'C1': 1, 'C2': 0, 'C3': 1}
    assert ac_solver(send_more_money) == \
           {'S': 9, 'M': 1, 'E': 5, 'N': 6, 'D': 7, 'O': 0, 'R': 8, 'Y': 2, 'C1': 1, 'C2': 1, 'C3': 0, 'C4': 1}


def test_ac_search_solver():
    assert ac_search_solver(csp_crossword) == {'one_across': 'has',
                                               'one_down': 'hold',
                                               'two_down': 'syntax',
                                               'three_across': 'land',
                                               'four_across': 'ant'} or {'one_across': 'bus',
                                                                         'one_down': 'buys',
                                                                         'two_down': 'search',
                                                                         'three_across': 'year',
                                                                         'four_across': 'car'}
    assert ac_search_solver(two_two_four) == {'T': 7, 'F': 1, 'W': 6, 'O': 5, 'U': 3, 'R': 0,
                                              'C1': 1, 'C2': 1, 'C3': 1} or \
           {'T': 9, 'F': 1, 'W': 2, 'O': 8, 'U': 5, 'R': 6, 'C1': 1, 'C2': 0, 'C3': 1}
    assert ac_search_solver(send_more_money) == {'S': 9, 'M': 1, 'E': 5, 'N': 6, 'D': 7, 'O': 0, 'R': 8, 'Y': 2,
                                                 'C1': 1, 'C2': 1, 'C3': 0, 'C4': 1}


def test_different_values_constraint():
    assert different_values_constraint('A', 1, 'B', 2)
    assert not different_values_constraint('A', 1, 'B', 1)


def test_flatten():
    sequence = [[0, 1, 2], [4, 5]]
    assert flatten(sequence) == [0, 1, 2, 4, 5]


def test_sudoku():
    h = Sudoku(easy1)
    assert backtracking_search(h, select_unassigned_variable=mrv, inference=forward_checking) is not None
    g = Sudoku(harder1)
    assert backtracking_search(g, select_unassigned_variable=mrv, inference=forward_checking) is not None


def test_make_arc_consistent():
    neighbors = parse_neighbors('A: B; B: ')
    domains = {'A': [0], 'B': [3]}
    constraints = lambda X, x, Y, y: x % 2 == 0 and (x + y) == 4

    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)
    csp.support_pruning()
    Xi = 'A'
    Xj = 'B'

    assert make_arc_consistent(Xi, Xj, csp) == []

    domains = {'A': [0], 'B': [4]}
    constraints = lambda X, x, Y, y: x % 2 == 0 and (x + y) == 4

    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)
    csp.support_pruning()
    Xi = 'A'
    Xj = 'B'

    assert make_arc_consistent(Xi, Xj, csp) == [0]

    domains = {'A': [0, 1, 2, 3, 4], 'B': [0, 1, 2, 3, 4]}
    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)
    csp.support_pruning()

    assert make_arc_consistent(Xi, Xj, csp) == [0, 2, 4]


def test_assign_value():
    neighbors = parse_neighbors('A: B; B: ')
    domains = {'A': [0, 1, 2, 3, 4], 'B': [0, 1, 2, 3, 4]}
    constraints = lambda X, x, Y, y: x % 2 == 0 and (x + y) == 4
    Xi = 'A'
    Xj = 'B'

    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)
    csp.support_pruning()

    assignment = {'A': 1}
    assert assign_value(Xi, Xj, csp, assignment) is None

    assignment = {'A': 2}
    assert assign_value(Xi, Xj, csp, assignment) == 2

    constraints = lambda X, x, Y, y: (x + y) == 4
    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)
    csp.support_pruning()

    assignment = {'A': 1}
    assert assign_value(Xi, Xj, csp, assignment) == 3


def test_no_inference():
    neighbors = parse_neighbors('A: B; B: ')
    domains = {'A': [0, 1, 2, 3, 4], 'B': [0, 1, 2, 3, 4, 5]}
    constraints = lambda X, x, Y, y: (x + y) < 8
    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)

    var = 'B'
    value = 3
    assignment = {'A': 1}
    assert no_inference(csp, var, value, assignment, None)


def test_mac():
    neighbors = parse_neighbors('A: B; B: ')
    domains = {'A': [0], 'B': [0]}
    constraints = lambda X, x, Y, y: x % 2 == 0
    var = 'B'
    value = 0
    assignment = {'A': 0}

    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)
    assert mac(csp, var, value, assignment, None)

    neighbors = parse_neighbors('A: B; B: ')
    domains = {'A': [0, 1, 2, 3, 4], 'B': [0, 1, 2, 3, 4]}
    constraints = lambda X, x, Y, y: x % 2 == 0 and (x + y) == 4 and y % 2 != 0
    var = 'B'
    value = 3
    assignment = {'A': 1}

    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)
    consistency, _ = mac(csp, var, value, assignment, None)
    assert not consistency

    constraints = lambda X, x, Y, y: x % 2 != 0 and (x + y) == 6 and y % 2 != 0
    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)
    _, consistency = mac(csp, var, value, assignment, None)
    assert consistency


def test_queen_constraint():
    assert queen_constraint(0, 1, 0, 1)
    assert queen_constraint(2, 1, 4, 2)
    assert not queen_constraint(2, 1, 3, 2)


def test_zebra():
    z = Zebra()
    algorithm = min_conflicts
    #  would take very long
    ans = algorithm(z, max_steps=10000)
    assert ans is None or ans == {'Red': 3, 'Yellow': 1, 'Blue': 2, 'Green': 5, 'Ivory': 4, 'Dog': 4, 'Fox': 1,
                                  'Snails': 3, 'Horse': 2, 'Zebra': 5, 'OJ': 4, 'Tea': 2, 'Coffee': 5, 'Milk': 3,
                                  'Water': 1, 'Englishman': 3, 'Spaniard': 4, 'Norwegian': 1, 'Ukranian': 2,
                                  'Japanese': 5, 'Kools': 1, 'Chesterfields': 2, 'Winston': 3, 'LuckyStrike': 4,
                                  'Parliaments': 5}

    #  restrict search space
    z.domains = {'Red': [3, 4], 'Yellow': [1, 2], 'Blue': [1, 2], 'Green': [4, 5], 'Ivory': [4, 5], 'Dog': [4, 5],
                 'Fox': [1, 2], 'Snails': [3], 'Horse': [2], 'Zebra': [5], 'OJ': [1, 2, 3, 4, 5],
                 'Tea': [1, 2, 3, 4, 5], 'Coffee': [1, 2, 3, 4, 5], 'Milk': [3], 'Water': [1, 2, 3, 4, 5],
                 'Englishman': [1, 2, 3, 4, 5], 'Spaniard': [1, 2, 3, 4, 5], 'Norwegian': [1],
                 'Ukranian': [1, 2, 3, 4, 5], 'Japanese': [1, 2, 3, 4, 5], 'Kools': [1, 2, 3, 4, 5],
                 'Chesterfields': [1, 2, 3, 4, 5], 'Winston': [1, 2, 3, 4, 5], 'LuckyStrike': [1, 2, 3, 4, 5],
                 'Parliaments': [1, 2, 3, 4, 5]}
    ans = algorithm(z, max_steps=10000)
    assert ans == {'Red': 3, 'Yellow': 1, 'Blue': 2, 'Green': 5, 'Ivory': 4, 'Dog': 4, 'Fox': 1, 'Snails': 3,
                   'Horse': 2, 'Zebra': 5, 'OJ': 4, 'Tea': 2, 'Coffee': 5, 'Milk': 3, 'Water': 1, 'Englishman': 3,
                   'Spaniard': 4, 'Norwegian': 1, 'Ukranian': 2, 'Japanese': 5, 'Kools': 1, 'Chesterfields': 2,
                   'Winston': 3, 'LuckyStrike': 4, 'Parliaments': 5}

iris_tests = [([5.0, 3.1, 0.9, 0.1], 0),
              ([5.1, 3.5, 1.0, 0.0], 0),
              ([4.9, 3.3, 1.1, 0.1], 0),
              ([6.0, 3.0, 4.0, 1.1], 1),
              ([6.1, 2.2, 3.5, 1.0], 1),
              ([5.9, 2.5, 3.3, 1.1], 1),
              ([7.5, 4.1, 6.2, 2.3], 2),
              ([7.3, 4.0, 6.1, 2.4], 2),
              ([7.0, 3.3, 6.1, 2.5], 2)]


def test_neural_net():
    iris = DataSet(name='iris')
    classes = ['setosa', 'versicolor', 'virginica']
    iris.classes_to_numbers(classes)
    n_samples, n_features = len(iris.examples), iris.target

    X, y = (np.array([x[:n_features] for x in iris.examples]),
            np.array([x[n_features] for x in iris.examples]))

    nnl_gd = NeuralNetworkLearner(iris, [4], l_rate=0.15, epochs=100, optimizer=stochastic_gradient_descent).fit(X, y)
    assert grade_learner(nnl_gd, iris_tests) > 0.7
    assert err_ratio(nnl_gd, iris) < 0.15

    nnl_adam = NeuralNetworkLearner(iris, [4], l_rate=0.001, epochs=200, optimizer=adam).fit(X, y)
    assert grade_learner(nnl_adam, iris_tests) > 0.7
    assert err_ratio(nnl_adam, iris) < 0.15


def test_perceptron():
    iris = DataSet(name='iris')
    classes = ['setosa', 'versicolor', 'virginica']
    iris.classes_to_numbers(classes)
    n_samples, n_features = len(iris.examples), iris.target

    X, y = (np.array([x[:n_features] for x in iris.examples]),
            np.array([x[n_features] for x in iris.examples]))

    pl_gd = PerceptronLearner(iris, l_rate=0.01, epochs=100, optimizer=stochastic_gradient_descent).fit(X, y)
    assert grade_learner(pl_gd, iris_tests) == 1
    assert err_ratio(pl_gd, iris) < 0.2

    pl_adam = PerceptronLearner(iris, l_rate=0.01, epochs=100, optimizer=adam).fit(X, y)
    assert grade_learner(pl_adam, iris_tests) == 1
    assert err_ratio(pl_adam, iris) < 0.2


def test_rnn():
    data = imdb.load_data(num_words=5000)

    train, val, test = keras_dataset_loader(data)
    train = (train[0][:1000], train[1][:1000])
    val = (val[0][:200], val[1][:200])

    rnn = SimpleRNNLearner(train, val)
    score = rnn.evaluate(test[0][:200], test[1][:200], verbose=False)
    assert score[1] >= 0.2


def test_autoencoder():
    iris = DataSet(name='iris')
    classes = ['setosa', 'versicolor', 'virginica']
    iris.classes_to_numbers(classes)
    inputs = np.asarray(iris.examples)

    al = AutoencoderLearner(inputs, 100)
    print(inputs[0])
    print(al.predict(inputs[:1]))

party = [
    {'Pizza': 'Yes', 'Soda': 'No', 'GOAL': True},
    {'Pizza': 'Yes', 'Soda': 'Yes', 'GOAL': True},
    {'Pizza': 'No', 'Soda': 'No', 'GOAL': False}]

animals_umbrellas = [
    {'Species': 'Cat', 'Rain': 'Yes', 'Coat': 'No', 'GOAL': True},
    {'Species': 'Cat', 'Rain': 'Yes', 'Coat': 'Yes', 'GOAL': True},
    {'Species': 'Dog', 'Rain': 'Yes', 'Coat': 'Yes', 'GOAL': True},
    {'Species': 'Dog', 'Rain': 'Yes', 'Coat': 'No', 'GOAL': False},
    {'Species': 'Dog', 'Rain': 'No', 'Coat': 'No', 'GOAL': False},
    {'Species': 'Cat', 'Rain': 'No', 'Coat': 'No', 'GOAL': False},
    {'Species': 'Cat', 'Rain': 'No', 'Coat': 'Yes', 'GOAL': True}]

conductance = [
    {'Sample': 'S1', 'Mass': 12, 'Temp': 26, 'Material': 'Cu', 'Size': 3, 'GOAL': 0.59},
    {'Sample': 'S1', 'Mass': 12, 'Temp': 100, 'Material': 'Cu', 'Size': 3, 'GOAL': 0.57},
    {'Sample': 'S2', 'Mass': 24, 'Temp': 26, 'Material': 'Cu', 'Size': 6, 'GOAL': 0.59},
    {'Sample': 'S3', 'Mass': 12, 'Temp': 26, 'Material': 'Pb', 'Size': 2, 'GOAL': 0.05},
    {'Sample': 'S3', 'Mass': 12, 'Temp': 100, 'Material': 'Pb', 'Size': 2, 'GOAL': 0.04},
    {'Sample': 'S4', 'Mass': 18, 'Temp': 100, 'Material': 'Pb', 'Size': 3, 'GOAL': 0.04},
    {'Sample': 'S4', 'Mass': 18, 'Temp': 100, 'Material': 'Pb', 'Size': 3, 'GOAL': 0.04},
    {'Sample': 'S5', 'Mass': 24, 'Temp': 100, 'Material': 'Pb', 'Size': 4, 'GOAL': 0.04},
    {'Sample': 'S6', 'Mass': 36, 'Temp': 26, 'Material': 'Pb', 'Size': 6, 'GOAL': 0.05}]


def r_example(Alt, Bar, Fri, Hun, Pat, Price, Rain, Res, Type, Est, GOAL):
    return {'Alt': Alt, 'Bar': Bar, 'Fri': Fri, 'Hun': Hun, 'Pat': Pat, 'Price': Price,
            'Rain': Rain, 'Res': Res, 'Type': Type, 'Est': Est, 'GOAL': GOAL}


restaurant = [
    r_example('Yes', 'No', 'No', 'Yes', 'Some', '$$$', 'No', 'Yes', 'French', '0-10', True),
    r_example('Yes', 'No', 'No', 'Yes', 'Full', '$', 'No', 'No', 'Thai', '30-60', False),
    r_example('No', 'Yes', 'No', 'No', 'Some', '$', 'No', 'No', 'Burger', '0-10', True),
    r_example('Yes', 'No', 'Yes', 'Yes', 'Full', '$', 'Yes', 'No', 'Thai', '10-30', True),
    r_example('Yes', 'No', 'Yes', 'No', 'Full', '$$$', 'No', 'Yes', 'French', '>60', False),
    r_example('No', 'Yes', 'No', 'Yes', 'Some', '$$', 'Yes', 'Yes', 'Italian', '0-10', True),
    r_example('No', 'Yes', 'No', 'No', 'None', '$', 'Yes', 'No', 'Burger', '0-10', False),
    r_example('No', 'No', 'No', 'Yes', 'Some', '$$', 'Yes', 'Yes', 'Thai', '0-10', True),
    r_example('No', 'Yes', 'Yes', 'No', 'Full', '$', 'Yes', 'No', 'Burger', '>60', False),
    r_example('Yes', 'Yes', 'Yes', 'Yes', 'Full', '$$$', 'No', 'Yes', 'Italian', '10-30', False),
    r_example('No', 'No', 'No', 'No', 'None', '$', 'No', 'No', 'Thai', '0-10', False),
    r_example('Yes', 'Yes', 'Yes', 'Yes', 'Full', '$', 'No', 'No', 'Burger', '30-60', True)]


def test_current_best_learning():
    examples = restaurant
    hypothesis = [{'Alt': 'Yes'}]
    h = current_best_learning(examples, hypothesis)
    values = [guess_value(e, h) for e in examples]

    assert values == [True, False, True, True, False, True, False, True, False, False, False, True]

    examples = animals_umbrellas
    initial_h = [{'Species': 'Cat'}]
    h = current_best_learning(examples, initial_h)
    values = [guess_value(e, h) for e in examples]

    assert values == [True, True, True, False, False, False, True]

    examples = party
    initial_h = [{'Pizza': 'Yes'}]
    h = current_best_learning(examples, initial_h)
    values = [guess_value(e, h) for e in examples]

    assert values == [True, True, False]


def test_version_space_learning():
    V = version_space_learning(party)
    results = []
    for e in party:
        guess = False
        for h in V:
            if guess_value(e, h):
                guess = True
                break

        results.append(guess)

    assert results == [True, True, False]
    assert [{'Pizza': 'Yes'}] in V


def test_minimal_consistent_det():
    assert minimal_consistent_det(party, {'Pizza', 'Soda'}) == {'Pizza'}
    assert minimal_consistent_det(party[:2], {'Pizza', 'Soda'}) == set()
    assert minimal_consistent_det(animals_umbrellas, {'Species', 'Rain', 'Coat'}) == {'Species', 'Rain', 'Coat'}
    assert minimal_consistent_det(conductance, {'Mass', 'Temp', 'Material', 'Size'}) == {'Temp', 'Material'}
    assert minimal_consistent_det(conductance, {'Mass', 'Temp', 'Size'}) == {'Mass', 'Temp', 'Size'}


A, B, C, D, E, F, G, H, I, x, y, z = map(expr, 'ABCDEFGHIxyz')

# knowledge base containing family relations
small_family = FOILContainer([expr("Mother(Anne, Peter)"),
                              expr("Mother(Anne, Zara)"),
                              expr("Mother(Sarah, Beatrice)"),
                              expr("Mother(Sarah, Eugenie)"),
                              expr("Father(Mark, Peter)"),
                              expr("Father(Mark, Zara)"),
                              expr("Father(Andrew, Beatrice)"),
                              expr("Father(Andrew, Eugenie)"),
                              expr("Father(Philip, Anne)"),
                              expr("Father(Philip, Andrew)"),
                              expr("Mother(Elizabeth, Anne)"),
                              expr("Mother(Elizabeth, Andrew)"),
                              expr("Male(Philip)"),
                              expr("Male(Mark)"),
                              expr("Male(Andrew)"),
                              expr("Male(Peter)"),
                              expr("Female(Elizabeth)"),
                              expr("Female(Anne)"),
                              expr("Female(Sarah)"),
                              expr("Female(Zara)"),
                              expr("Female(Beatrice)"),
                              expr("Female(Eugenie)")])

smaller_family = FOILContainer([expr("Mother(Anne, Peter)"),
                                expr("Father(Mark, Peter)"),
                                expr("Father(Philip, Anne)"),
                                expr("Mother(Elizabeth, Anne)"),
                                expr("Male(Philip)"),
                                expr("Male(Mark)"),
                                expr("Male(Peter)"),
                                expr("Female(Elizabeth)"),
                                expr("Female(Anne)")])

# target relation
target = expr('Parent(x, y)')

# positive examples of target
examples_pos = [{x: expr('Elizabeth'), y: expr('Anne')},
                {x: expr('Elizabeth'), y: expr('Andrew')},
                {x: expr('Philip'), y: expr('Anne')},
                {x: expr('Philip'), y: expr('Andrew')},
                {x: expr('Anne'), y: expr('Peter')},
                {x: expr('Anne'), y: expr('Zara')},
                {x: expr('Mark'), y: expr('Peter')},
                {x: expr('Mark'), y: expr('Zara')},
                {x: expr('Andrew'), y: expr('Beatrice')},
                {x: expr('Andrew'), y: expr('Eugenie')},
                {x: expr('Sarah'), y: expr('Beatrice')},
                {x: expr('Sarah'), y: expr('Eugenie')}]

# negative examples of target
examples_neg = [{x: expr('Anne'), y: expr('Eugenie')},
                {x: expr('Beatrice'), y: expr('Eugenie')},
                {x: expr('Mark'), y: expr('Elizabeth')},
                {x: expr('Beatrice'), y: expr('Philip')}]


def test_tell():
    """
    adds in the knowledge base a sentence
    """
    smaller_family.tell(expr("Male(George)"))
    smaller_family.tell(expr("Female(Mum)"))
    assert smaller_family.ask(expr("Male(George)")) == {}
    assert smaller_family.ask(expr("Female(Mum)")) == {}
    assert not smaller_family.ask(expr("Female(George)"))
    assert not smaller_family.ask(expr("Male(Mum)"))


def test_extend_example():
    """
    Create the extended examples of the given clause. 
    (The extended examples are a set of examples created by extending example 
    with each possible constant value for each new variable in literal.)
    """
    assert len(list(small_family.extend_example({x: expr('Andrew')}, expr('Father(x, y)')))) == 2
    assert len(list(small_family.extend_example({x: expr('Andrew')}, expr('Mother(x, y)')))) == 0
    assert len(list(small_family.extend_example({x: expr('Andrew')}, expr('Female(y)')))) == 6


def test_new_literals():
    assert len(list(small_family.new_literals([expr('p'), []]))) == 8
    assert len(list(small_family.new_literals([expr('p & q'), []]))) == 20


def test_new_clause():
    """
    Finds the best clause to add in the set of clauses.
    """
    clause = small_family.new_clause([examples_pos, examples_neg], target)[0][1]
    assert len(clause) == 1 and (clause[0].op in ['Male', 'Female', 'Father', 'Mother'])


def test_choose_literal():
    """
    Choose the best literal based on the information gain
    """
    literals = [expr('Father(x, y)'), expr('Father(x, y)'), expr('Mother(x, y)'), expr('Mother(x, y)')]
    examples_pos = [{x: expr('Philip')}, {x: expr('Mark')}, {x: expr('Peter')}]
    examples_neg = [{x: expr('Elizabeth')}, {x: expr('Sarah')}]
    assert small_family.choose_literal(literals, [examples_pos, examples_neg]) == expr('Father(x, y)')
    literals = [expr('Father(x, y)'), expr('Father(y, x)'), expr('Male(x)')]
    examples_pos = [{x: expr('Philip')}, {x: expr('Mark')}, {x: expr('Andrew')}]
    examples_neg = [{x: expr('Elizabeth')}, {x: expr('Sarah')}]
    assert small_family.choose_literal(literals, [examples_pos, examples_neg]) == expr('Father(x,y)')


def test_gain():
    """
    Calculates the utility of each literal, based on the information gained. 
    """
    gain_father = small_family.gain(expr('Father(x,y)'), [examples_pos, examples_neg])
    gain_male = small_family.gain(expr('Male(x)'), [examples_pos, examples_neg])
    assert round(gain_father, 2) == 2.49
    assert round(gain_male, 2) == 1.16


def test_update_examples():
    """Add to the kb those examples what are represented in extended_examples
        List of omitted examples is returned.
    """
    extended_examples = [{x: expr("Mark"), y: expr("Peter")},
                         {x: expr("Philip"), y: expr("Anne")}]

    uncovered = smaller_family.update_examples(target, examples_pos, extended_examples)
    assert {x: expr("Elizabeth"), y: expr("Anne")} in uncovered
    assert {x: expr("Anne"), y: expr("Peter")} in uncovered
    assert {x: expr("Philip"), y: expr("Anne")} not in uncovered
    assert {x: expr("Mark"), y: expr("Peter")} not in uncovered


def test_foil():
    """
    Test the FOIL algorithm, when target is  Parent(x,y)
    """
    clauses = small_family.foil([examples_pos, examples_neg], target)
    assert len(clauses) == 2 and \
           ((clauses[0][1][0] == expr('Father(x, y)') and clauses[1][1][0] == expr('Mother(x, y)')) or
            (clauses[1][1][0] == expr('Father(x, y)') and clauses[0][1][0] == expr('Mother(x, y)')))

    target_g = expr('Grandparent(x, y)')
    examples_pos_g = [{x: expr('Elizabeth'), y: expr('Peter')},
                      {x: expr('Elizabeth'), y: expr('Zara')},
                      {x: expr('Elizabeth'), y: expr('Beatrice')},
                      {x: expr('Elizabeth'), y: expr('Eugenie')},
                      {x: expr('Philip'), y: expr('Peter')},
                      {x: expr('Philip'), y: expr('Zara')},
                      {x: expr('Philip'), y: expr('Beatrice')},
                      {x: expr('Philip'), y: expr('Eugenie')}]
    examples_neg_g = [{x: expr('Anne'), y: expr('Eugenie')},
                      {x: expr('Beatrice'), y: expr('Eugenie')},
                      {x: expr('Elizabeth'), y: expr('Andrew')},
                      {x: expr('Elizabeth'), y: expr('Anne')},
                      {x: expr('Elizabeth'), y: expr('Mark')},
                      {x: expr('Elizabeth'), y: expr('Sarah')},
                      {x: expr('Philip'), y: expr('Anne')},
                      {x: expr('Philip'), y: expr('Andrew')},
                      {x: expr('Anne'), y: expr('Peter')},
                      {x: expr('Anne'), y: expr('Zara')},
                      {x: expr('Mark'), y: expr('Peter')},
                      {x: expr('Mark'), y: expr('Zara')},
                      {x: expr('Andrew'), y: expr('Beatrice')},
                      {x: expr('Andrew'), y: expr('Eugenie')},
                      {x: expr('Sarah'), y: expr('Beatrice')},
                      {x: expr('Mark'), y: expr('Elizabeth')},
                      {x: expr('Beatrice'), y: expr('Philip')},
                      {x: expr('Peter'), y: expr('Andrew')},
                      {x: expr('Zara'), y: expr('Mark')},
                      {x: expr('Peter'), y: expr('Anne')},
                      {x: expr('Zara'), y: expr('Eugenie')}]

    clauses = small_family.foil([examples_pos_g, examples_neg_g], target_g)
    assert len(clauses[0]) == 2
    assert clauses[0][1][0].op == 'Parent'
    assert clauses[0][1][0].args[0] == x
    assert clauses[0][1][1].op == 'Parent'
    assert clauses[0][1][1].args[1] == y


def speak(text):
    engine.say(text)
    engine.runAndWait()

def takeName():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print('Menunggu nama....')
        r.pause_threshold = 1
        audio = r.listen(source)
    try:
        print("Mengenali nama.....")
        name = r.recognize_google(audio, language='en-in')
        print("Nama Anda: {} \n".format(name))
    except sr.UnknownValueError:
        print("Ucapkan lagi....")
        return "None"
    return name

def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print('Listening....')
        r.pause_threshold = 1
        audio = r.listen(source)
    try:
        print("Recognizing.....")
        query = r.recognize_google(audio, language='en-in')
        print(f"{name} Said: {query} \n")
        add_user_data(name, query)  # Simpan data pengguna ke database
    except sr.UnknownValueError:
        print("Say That Again....")
        return "None"
    except sr.RequestError as e:
        print("Could not request results. Check your network connection.")
        return "None"
    return query

if __name__ == '__main__':
    initialize_database()
    name = takeName()
    speak(f"Selamat datang, {name}! Apa yang bisa saya bantu?")
    while True:
        query = takeCommand().lower()
        ans = Reply(query)
        print(ans)
        speak(ans)
        if 'open youtube' in query:
            webbrowser.open("www.youtube.com")
        elif 'open google' in query:
            webbrowser.open("www.google.com")
        elif 'search' in query:
            # Parsing query untuk mencari kata kunci pencarian
            search_query = query.replace('search', '').strip()
            # Menjalankan fungsi pencarian dari modul yang diimpor
            search_results = search_in_database(search_query)
            print("Search Results:", search_results)
            speak("Here are the search results:")
            for result in search_results:
                print(result)
                speak(result)
        elif 'bye' in query:
            break
    db_connection.close()