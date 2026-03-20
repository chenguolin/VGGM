#!/usr/bin/env bash
# ── Pretty Bash prompt ────────────────────────────────────────────────────────
# Source this file from ~/.bashrc:
#   source /path/to/projects/VGGM/settings/bashrc.sh

# Colors (256-color safe, degrade gracefully on basic terminals)
_clr_reset='\[\e[0m\]'
_clr_bold='\[\e[1m\]'
_clr_red='\[\e[38;5;203m\]'       # soft red   – non-zero exit code
_clr_green='\[\e[38;5;114m\]'     # soft green – zero exit code
_clr_yellow='\[\e[38;5;220m\]'    # gold       – conda env
_clr_blue='\[\e[38;5;75m\]'       # sky blue   – working dir
_clr_purple='\[\e[38;5;183m\]'    # lavender   – git branch
_clr_gray='\[\e[38;5;245m\]'      # gray       – timestamp / separators

# Git branch in prompt
_git_branch() {
    local branch
    branch=$(git symbolic-ref --short HEAD 2>/dev/null) \
        || branch=$(git rev-parse --short HEAD 2>/dev/null)
    [ -n "$branch" ] && printf ' ⎇ %s' "$branch"   # git branch symbol
}

# Conda env label (skip "base" to reduce noise)
_conda_env() {
    local env="${CONDA_DEFAULT_ENV:-}"
    [ -n "$env" ] && [ "$env" != "base" ] && printf '(%s) ' "$env"
}

# The prompt itself – built fresh before each command via PROMPT_COMMAND
_set_ps1() {
    local exit_code=$?
    local tick
    if [ $exit_code -eq 0 ]; then
        tick="${_clr_green}✔${_clr_reset}"
    else
        tick="${_clr_red}✘ ${exit_code}${_clr_reset}"
    fi

    local ts="${_clr_gray}$(date +'%H:%M:%S')${_clr_reset}"
    local conda="${_clr_yellow}$(_conda_env)${_clr_reset}"
    local dir="${_clr_bold}${_clr_blue}\w${_clr_reset}"
    local git="${_clr_purple}$(_git_branch)${_clr_reset}"

    # Line 1: timestamp  tick  conda-env  dir  git-branch
    # Line 2: prompt character
    PS1="${_clr_gray}┌─${_clr_reset} ${ts}  ${tick}  ${conda}${dir}${git}\n${_clr_gray}└─${_clr_reset} ${_clr_bold}\$${_clr_reset} "
}
PROMPT_COMMAND='_set_ps1'
# ── End pretty prompt ──────────────────────────────────────────────────────────

# ── History search with arrow keys ────────────────────────────────────────────
# Type a few characters then press ↑/↓ to search history matching the prefix
# Guard: `bind` and `shopt` are bash-only builtins; skip silently in zsh/sh
if [ -n "$BASH_VERSION" ]; then
    # Arrow-key history search (interactive shells only)
    if [[ $- == *i* ]]; then
        bind '"\e[A": history-search-backward'
        bind '"\e[B": history-search-forward'
    fi
    shopt -s histappend
fi
# Larger history & no duplicates
export HISTSIZE=10000
export HISTFILESIZE=20000
export HISTCONTROL=ignoreboth:erasedups
# ── End history config ────────────────────────────────────────────────────────
