# Building a Git Commit Message Generator CLI with Ollama

In this guide, we'll build a command-line tool that generates well-structured commit messages by analyzing your git changes. This practical example demonstrates how to use Outlines' [Ollama integration](features/models/ollama.md) to create structured outputs that follow conventional commit standards.

## What We'll Build

Our CLI tool will:

- Analyze staged git changes
- Generate commit messages following the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification
- Provide structured output with type, scope, description, and body
- Support different commit types (feat, fix, docs, etc.)

## Prerequisites

Before starting, ensure you have Ollama installed and running locally.

## Setting Up the Project

First, let's create a new project and install the required dependencies:

```shell
mkdir git-commit-cli
cd git-commit-cli
uv init
uv add outlines ollama click gitpython
```

We're using:
- `outlines` for structured generation with Ollama
- `ollama` for the Ollama client
- `click` for building the CLI interface
- `gitpython` for interacting with git repositories

## Step 1: Understanding Structured Generation with Ollama

Before we dive into building our CLI, let's understand how Outlines works with Ollama for structured generation.

### How Outlines Integrates with Ollama

Outlines provides a simple interface to Ollama models through the `from_ollama` function. The key feature we'll use is structured generation:

1. You create an Ollama client
2. You pass the client to Outlines along with a model name
3. You define a Pydantic model that describes your desired output structure
4. Outlines ensures the model generates JSON that matches your schema

This is particularly powerful because it guarantees that the output will always match your expected format, making it perfect for building reliable CLI tools.

## Step 2: Building Our First Structured CLI

Now comes the powerful part - using structured generation to ensure our commit messages follow a consistent format. This is where Outlines shines with Ollama.

### Understanding Structured Generation

When you pass a Pydantic model to an Ollama model through Outlines, it:

1. Converts your Pydantic model to a JSON schema
2. Tells Ollama to generate JSON that matches this schema
3. Returns the JSON string that you can validate

Let's build this step by step:

### First, Define Our Data Model

```python
# commit_cli.py
from pydantic import BaseModel, Field
from typing import Optional, Literal

# This model defines the structure of our commit message
class CommitMessage(BaseModel):
    """Structured commit message following Conventional Commits."""
    type: Literal["feat", "fix", "docs", "style", "refactor", "test", "chore"]
    scope: Optional[str] = Field(None, description="Component or area affected")
    description: str = Field(..., description="Short summary (max 72 chars)")
    body: Optional[str] = Field(None, description="Detailed explanation")
    breaking: bool = Field(False, description="Is this a breaking change?")
```

### Now, Let's Use Structured Generation

```python
# commit_cli.py (continued)
import click
import ollama
import outlines
from outlines import Template


@click.command()
@click.option('--model', default='llama3.2:3b', help='Ollama model to use')
@click.option('--port', default=11434, help='Ollama server port (default: 11434)')
def generate_commit(model, port):
    """Generate a structured commit message."""
    click.echo("Structured Commit Message Generator")
    click.echo("=" * 40)
    
    # Get change description from user
    changes = click.prompt("Describe your changes", type=str)
    
    client = ollama.Client(host=f"http://localhost:{port}")
    llm = outlines.from_ollama(client, model)

    template = Template("""
Based on these changes:
{{ changes }}

Generate a conventional commit message with:
- type: The commit type (feat, fix, docs, etc.)
- scope: The component affected (optional)
- description: A short summary
- body: Detailed explanation (optional)
- breaking: Whether this is a breaking change
""")
    prompt = template(changes=changes)
    
    click.echo(f"\nGenerating with {model}...")
    
    response = llm(prompt, CommitMessage)
    commit = CommitMessage.model_validate_json(response)
    
    message = format_commit_message(commit)
    click.echo("\nFormatted Commit Message:")
    click.echo("-" * 40)
    click.echo(message)
    click.echo("-" * 40)


def format_commit_message(commit: CommitMessage) -> str:
    """Format the structured data into a conventional commit message."""
    message = f"{commit.type}"
    if commit.scope:
        message += f"({commit.scope})"
    message += f": {commit.description}"
    
    if commit.body:
        message += f"\n\n{commit.body}"
    
    if commit.breaking:
        message += "\n\nBREAKING CHANGE: This commit introduces breaking changes"
    
    return message


if __name__ == '__main__':
    generate_commit()
```

## Step 3: Integrating with Git

Now let's build upon the previous implementation to analyze actual Git changes.

### Understanding Git Integration

First, let's understand what we need to do:
1. Read the current Git repository
2. Find staged changes
3. Extract file changes and diffs
4. Pass this information to our LLM

### Step 3.1: Basic Git Analysis

Let's start by building a simple function to read Git changes:

```python
# git_basics.py
import git
from pathlib import Path


def check_git_repository(path: str = ".") -> bool:
    """Check if the current directory is a git repository."""
    try:
        repo = git.Repo(path)
        print(f"✓ Found git repository at: {repo.working_dir}")
        print(f"  Current branch: {repo.active_branch}")
        print(f"  Has staged changes: {len(repo.index.diff('HEAD')) > 0}")
        return True
    except git.InvalidGitRepositoryError:
        print("✗ Not a git repository")
        return False


if __name__ == "__main__":
    check_git_repository()
```

### Step 3.2: Analyzing Staged Changes

Now let's build a function to analyze what's been staged:

```python
# analyze_changes.py
import git
from typing import List, Dict

def get_staged_files(repo_path: str = ".") -> List[Dict]:
    """Get information about staged files."""
    repo = git.Repo(repo_path)
    staged_files = []
    
    staged_diff = repo.index.diff("HEAD")
    
    for item in staged_diff:
        if item.change_type == 'A':
            status = "Added"
        elif item.change_type == 'M':
            status = "Modified"
        elif item.change_type == 'D':
            status = "Deleted"
        elif item.change_type == 'R':
            status = "Renamed"
        else:
            status = item.change_type
        
        file_info = {
            "filename": item.a_path or item.b_path,
            "status": status,
            "change_type": item.change_type
        }
        staged_files.append(file_info)
    
    return staged_files

if __name__ == "__main__":
    files = get_staged_files()
    for f in files:
        print(f"{f['status']:10} {f['filename']}")
```

### Step 3.3: Building the Complete Git Commit Generator

Now let's put it all together:

```python
# git_commit_generator.py
import click
import ollama
import outlines
from outlines import Template
from pydantic import BaseModel, Field
from typing import Optional, Literal, List
import git
from pathlib import Path


class FileChange(BaseModel):
    """Represents a single file change."""
    filename: str
    additions: int
    deletions: int
    status: str  # Added, Modified, Deleted, Renamed


class CommitMessage(BaseModel):
    """Structured commit message following Conventional Commits."""
    type: Literal["feat", "fix", "docs", "style", "refactor", "test", "chore", "perf", "ci"]
    scope: Optional[str] = Field(None, description="Component or area affected")
    description: str = Field(..., description="Short summary (max 72 chars)")
    body: Optional[str] = Field(None, description="Detailed explanation")
    breaking: bool = Field(False, description="Is this a breaking change?")
    issues: Optional[List[str]] = Field(None, description="Related issue numbers")


def analyze_git_changes(repo_path: str = ".") -> tuple[str, List[FileChange]]:
    """
    Analyze staged changes in the git repository.
    Returns a summary text and list of file changes.
    """
    try:
        repo = git.Repo(repo_path)
    except git.InvalidGitRepositoryError:
        raise click.ClickException("Not a git repository!")
    
    if repo.bare:
        raise click.ClickException("Cannot analyze bare repository")
    
    staged_files = []
    diff_summary = []
    
    if repo.head.is_valid():
        staged_diff = repo.index.diff("HEAD")
    else:
        staged_diff = repo.index.diff(None)
        click.echo("Note: This will be the first commit")
    
    for item in staged_diff:
        # Map change types to human-readable status
        status_map = {
            'A': "Added",
            'M': "Modified", 
            'D': "Deleted",
            'R': "Renamed"
        }
        status = status_map.get(item.change_type, item.change_type)
        
        additions = 0
        deletions = 0
        if item.diff:
            diff_lines = item.diff.decode('utf-8', errors='ignore').split('\n')
            additions = sum(1 for line in diff_lines if line.startswith('+'))
            deletions = sum(1 for line in diff_lines if line.startswith('-'))
        
        file_change = FileChange(
            filename=item.a_path or item.b_path,
            additions=additions,
            deletions=deletions,
            status=status
        )
        staged_files.append(file_change)
        
        diff_summary.append(f"{status}: {file_change.filename} "
                          f"(+{additions}/-{deletions})")
    
    unstaged = repo.index.diff(None)
    if unstaged:
        click.echo(click.style(
            f"⚠️  You have {len(unstaged)} unstaged file(s)", 
            fg="yellow"
        ))
    
    return "\n".join(diff_summary), staged_files


@click.command()
@click.option('--model', default='llama3.2:3b', help='Ollama model to use')
@click.option('--repo', default='.', help='Path to git repository')
@click.option('--dry-run', is_flag=True, help='Show message without committing')
@click.option('--port', default=11434, help='Ollama server port (default: 11434)')
def generate_commit(model, repo, dry_run, port):
    """Generate a commit message based on staged git changes."""
    click.echo(click.style("Git Commit Message Generator", bold=True))
    click.echo("=" * 40)
    
    repo_path = Path(repo).absolute()
    click.echo(f"Repository: {repo_path}")
    
    try:
        # Step 1: Analyze the repository
        click.echo("\n📋 Analyzing staged changes...")
        summary, files = analyze_git_changes(repo)
        
        if not files:
            click.echo(click.style("\n❌ No staged changes found!", fg="red"))
            click.echo("\nTo stage changes, use:")
            click.echo("  git add <file>     # Stage specific file")
            click.echo("  git add .          # Stage all changes")
            return
        
        # Step 2: Generate the commit message
        client = ollama.Client(host=f"http://localhost:{port}")
        llm = outlines.from_ollama(client, model)
        
        template = Template("""
Analyze these git changes and generate a conventional commit message.

Files changed:
{{ file_summary }}

Instructions:
- Choose the most appropriate type (feat, fix, docs, etc.)
- Identify the scope if changes are focused on a specific component
- Write a clear, concise description (imperative mood, max 72 chars)
- Add a body if the changes need more explanation
- Mark as breaking if there are backwards-incompatible changes
- Extract any issue numbers mentioned in the file names or paths
""")
        prompt = template(file_summary=summary)
        response = llm(prompt, CommitMessage)
        commit = CommitMessage.model_validate_json(response)
        
        final_message = format_commit_message(commit)
        
        click.echo("\n" + click.style("Generated Commit Message:", bold=True))
        click.echo("─" * 50)
        click.echo(final_message)
        click.echo("─" * 50)
        
        # Commit if not dry-run
        if not dry_run:
            if click.confirm("\n✅ Use this commit message?"):
                repo_obj = git.Repo(repo)
                repo_obj.index.commit(final_message)
                click.echo(click.style("✨ Changes committed successfully!", 
                                     fg="green", bold=True))
            else:
                click.echo("Commit cancelled.")
        else:
            click.echo("\n(Dry run - no commit made)")
        
    except git.InvalidGitRepositoryError:
        click.echo(click.style("\n❌ Error: Not a git repository", fg="red"))
        click.echo("Initialize a repository with: git init")
    except Exception as e:
        click.echo(click.style(f"\n❌ Error: {str(e)}", fg="red"))


def format_commit_message(commit: CommitMessage) -> str:
    """Format the structured commit data into a proper commit message."""
    lines = []
    
    first_line = f"{commit.type}"
    if commit.scope:
        first_line += f"({commit.scope})"
    first_line += f": {commit.description}"
    lines.append(first_line)
    
    if commit.body:
        lines.extend(["", commit.body])
    
    if commit.breaking or commit.issues:
        lines.append("")
        
        if commit.breaking:
            lines.append("BREAKING CHANGE: See commit body for details")
        
        if commit.issues:
            for issue in commit.issues:
                lines.append(f"Fixes #{issue}")
    
    return "\n".join(lines)


if __name__ == '__main__':
    generate_commit()
```

### Try It Out

```shell
# In a git repository with staged changes:
uv run python git_commit_generator.py

# Preview without committing:
uv run python git_commit_generator.py --dry-run

# Use a different model:
uv run python git_commit_generator.py --model codellama
```

## Running the Tool

Save the complete script and run it in your git repository:

```shell
# Basic usage
uv run python git_commit_generator.py

# Use a different model
uv run python git_commit_generator.py --model codellama

# Dry run to preview without committing
uv run python git_commit_generator.py --dry-run

# Use Ollama on a different port
uv run python git_commit_generator.py --port 8080

# Combine options
uv run python git_commit_generator.py --model codellama --port 8080 --dry-run
```

## Packaging as a Standalone Tool

To make this tool easily installable and usable system-wide:

```python
# pyproject.toml
[project]
name = "git-commit-ai"
version = "0.1.0"
description = "AI-powered git commit message generator"
dependencies = [
    "outlines>=0.1.0",
    "ollama>=0.1.0",
    "click>=8.0",
    "gitpython>=3.0",
    "pydantic>=2.0"
]

[project.scripts]
git-commit-ai = "git_commit_generator:generate_commit"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

Install it:
```shell
uv pip install -e .
```

Now you can use it anywhere:
```shell
git-commit-ai --help
```


## Next Steps

You can extend this tool further by:

1. **Configuration Files**: Add support for user preferences
   - Default model selection
   - Custom commit types for your team
   - Formatting preferences (emoji prefixes, statistics)

2. **Enhanced Validation**: Use Pydantic validators
   - Ensure descriptions follow conventions (lowercase, no period)
   - Validate scope against project components
   - Check description length limits

3. **Interactive Mode**: Allow users to edit generated messages
   - Prompt for each field individually
   - Preview and modify before committing
   - Save frequently used messages as templates

4. **Pattern Analysis**: Detect common patterns in changes
   - Suggest scope based on file paths (e.g., "docs" for README changes)
   - Identify test files and suggest "test" type
   - Recognize CI/CD files for appropriate categorization

5. **Integration Features**:
   - **Issue Tracking**: Auto-extract issue numbers from branch names or comments
   - **Pre-commit Hooks**: Validate messages before allowing commits
   - **Team Conventions**: Load organization-specific rules and templates

6. **Advanced Git Features**:
   - Analyze diff content, not just file names
   - Support for interactive rebase message generation
   - Generate PR descriptions from multiple commits

7. **Performance Optimizations**:
   - Cache similar commit patterns for faster generation
   - Stream responses for better user experience
   - Batch process multiple commits

8. **Multi-language Support**: Generate messages in different languages based on team preferences

## Conclusion

We've built a practical CLI tool that showcases Outlines' Ollama integration for structured generation. The tool demonstrates:

- Clean integration with local Ollama models
- Structured output using Pydantic models
- Real-world application with git integration
- Proper error handling and user experience
- Extensibility for team workflows

This pattern can be applied to many other CLI tools where you need structured, consistent output from language models - from code review assistants to documentation generators.
