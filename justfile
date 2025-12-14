# List available commands
default:
    @just --list

# Run TP1 demo (Single Object Tracking)
demo-tp1:
    cd tp1 && python main.py

# Run TP2 demo (IoU Tracker)
demo-tp2:
    python tp2/main.py

# Run TP3 demo (Kalman-Guided IoU Tracker)
demo-tp3:
    python tp3/main.py

# Run TP4 demo (Appearance-Aware Tracker)
demo-tp4:
    python tp4/main.py

# Compile the project report
report:
    typst compile report/main.typ

# Run all demos and compile report
all: demo-tp1 demo-tp2 demo-tp3 demo-tp4 report
