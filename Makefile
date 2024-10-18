# Define the absolute path to the TeamB directory
TEAM_B_PATH := $(shell realpath TeamB)

# Makefile

# Define paths to other Makefiles relative to the TeamB directory
MAKEFILE1 = $(TEAM_B_PATH)/Code/Bash/MF1
MAKEFILE2 = $(TEAM_B_PATH)/Code/Bash/MF2
MAKEFILE3 = $(TEAM_B_PATH)/Code/Bash/MF3

# Default target
all: run_part1 run_part2 run_part3

# Targets to run each part
run_part1:
	$(MAKE) -f $(MAKEFILE1)

run_part2:
	$(MAKE) -f $(MAKEFILE2)

run_part3:
	$(MAKE) -f $(MAKEFILE3)

# Optional: Target to clean up all parts
clean:
	$(MAKE) -C $(TEAM_B_PATH)/Code/Bash/MF1 clean
	$(MAKE) -C $(TEAM_B_PATH)/Code/Bash/MF2 clean
	$(MAKE) -C $(TEAM_B_PATH)/Code/Bash/MF3 clean

# Optional: Provide a .PHONY rule for non-file targets
.PHONY: all run_part1 run_part2 run_part3 clean
