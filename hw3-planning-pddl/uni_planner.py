import argparse
from pathlib import Path

from unified_planning import Environment
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner


def parse_arguments() -> argparse.Namespace:
    """Parses the command line arguments."""
    parser = argparse.ArgumentParser(description="Generate sailing planning instance")
    parser.add_argument(
        "--domain_file", type=str, required=True, help="Path to the domain file"
    )
    parser.add_argument(
        "--problem_file", type=str, required=True, help="Path to the problem file"
    )

    args = parser.parse_args()
    return args


def run_planner(domain_file: Path, problem_file: Path):
    env = Environment()
    reader = PDDLReader(env)
    problem = reader.parse_problem(domain_file, problem_file)

    with OneshotPlanner(name="pyperplan") as planner:
        result = planner.solve(problem)

    print("Status:", result.status)
    if result.plan is not None:
        print("Plan found:")
        for action_instance in result.plan.actions:
            print(action_instance)
    else:
        print("No plan found.")


def main():
    args = parse_arguments()
    run_planner(
        Path(args.domain_file),
        Path(args.problem_file),
    )


if __name__ == "__main__":
    main()
