FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy dependency files
COPY pyproject.toml ./

# Configure Poetry to not create virtualenv (we're in a container)
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-interaction --no-ansi --no-root

# Copy source code
COPY . .

# Install the package
RUN poetry install --no-interaction --no-ansi

# Create workspace directory
RUN mkdir -p /workspace

# Set environment variables
ENV JIRA_AGENT_WORKSPACE_DIR=/workspace
ENV PYTHONUNBUFFERED=1

# Default command
ENTRYPOINT ["jira-agent"]
CMD ["--help"]
