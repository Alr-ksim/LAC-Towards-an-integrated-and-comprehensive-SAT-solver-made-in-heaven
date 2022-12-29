## LAC_SAT Solver

### Environment
```
OS: Window 10
Python: 3.10.6
Packeges:
    pip install -r requirement.txt
```

### Solving method
* Put the input *.cnf* file in the *\examples* folder.
* Run the project using terminal:
```
    python main.py -i examples/test.cnf -d CHB -r EXP3
```
* The args are as followed:
    * -i: The path to the input *.cnf* file
    * -d: The decision method, option [VSIDS, CHB, LRB]
    * -r: The restart strategy, option [Nothing, EXP3, UCB]