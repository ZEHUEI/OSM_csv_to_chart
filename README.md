# Membership Chart FastAPI

FastAPI backend that accepts a CSV upload and returns the generated chart as a PNG.

## Files

- `main.py` - API endpoints
- `chart_logic.py` - chart generation
- `requirements.txt` - Python dependencies
- `render.yaml` - Render deployment configuration

## Run locally

Create/activate a virtual environment if desired, then:

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Use `POST /generate-chart`, click **Try it out**, upload your CSV, and execute.

## Test with curl

```bash
curl -X POST \
  -F "file=@dummy_membership_data.csv" \
  http://127.0.0.1:8000/generate-chart \
  --output chart.png
```

## CSV structure

Column 1's header becomes the chart title.

Example:

```csv
Membership,2025H1 (ACTUAL),2026H1 (Actual),2026H1 (Target)
Label,"5,714,747","4,754,265","7,070,434"
Publishing,"1,477,058","1,164,010","1,867,902"
Overall,"7,191,805","5,918,275","8,938,336"
O E,"5,372,220","",""
O W,"1,819,585","",""
P E,"1,440,870","",""
P W,"36,188","",""
L E,"3,931,350","",""
L W,"1,783,397","",""
```

The `E/W` rows are helper rows and do not appear as separate categories.

## Deploy to Render

Push this folder to GitHub.

If Render detects `render.yaml`, create a Blueprint from the repository.

Or create a Web Service manually with:

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

After deployment:

```text
https://YOUR-SERVICE.onrender.com/docs
```

The frontend should send a multipart/form-data POST request with the CSV under the field name `file`.
