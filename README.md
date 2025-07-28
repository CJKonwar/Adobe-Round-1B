# Execution Instruction

### - **approach_explanation.md CONTAINS THE METHODOLOGY OF OUR SOLUTION**

Docker Setup Instructions for Round 1B Solution

## Prerequisites
Before building and running the Docker container, follow these steps:

---

### Step 1: Install the required libary for downloading the model


```python
pip install sentence-transformers
```

### Step 2: Download Required Models
First, run the model download script to cache the required models locally:

```python
python downloadModel.py
```

---

### Step 2: Setup Directory Structure
Create the required directory structure on your host machine:

```python
## Create input directory and add your PDF files
mkdir -p input

## Copy your PDF files to the input directory
cp your_pdf_files*.pdf input/

## Create output directory (this will receive the generated results)
mkdir -p output
```

## Create your input JSON file with persona and job-to-be-done
## Example: input.json, test_input.json, etc.


---

### Step 3: Build Docker Image
Build the Docker image using the following command:

```python
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier2 .
```

---

### Step 4: Run Docker Container
Run the container with proper volume mounts:

```python
docker run --rm -it \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/input.json:/app/input.json \
  --network none \
  mysolutionname:somerandomidentifier2
```

---

### Custom Directory Names
If you're using different directory/file names, modify the command accordingly:

```python
docker run --rm -it \
  -v $(pwd)/[your_input_directory_name]:/app/input \
  -v $(pwd)/[your_output_directory_name]:/app/output \
  -v $(pwd)/[your_input_json_filename]:/app/input.json \
  --network none \
  [your docker image name]
```
