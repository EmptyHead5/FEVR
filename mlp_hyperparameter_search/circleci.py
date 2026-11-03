import os
import re
import requests
import pandas as pd
from dotenv import load_dotenv

from pathlib import Path

csv_folder = Path(__file__).resolve().parent / "csv"
csv_folder.mkdir(exist_ok=True)
csv_file = csv_folder / "circleci_hourly_trading_history.csv"

#study from https://circleci.com/docs/api/v2/
load_dotenv()

token = os.getenv("CIRCLECI_TOKEN")
headers = {"Circle-Token": token}

results = []

url = "https://circleci.com/api/v2/project/github/EmptyHead5/FEVR/pipeline"
pipelines = []




def getAllPipelines():
    pipelines = []
    params = {}

    while True:
        print(f"Fetching data from: {url}")
        response = requests.get(
            url, headers=headers, timeout=10, params=params
        )
        response.raise_for_status()
        data = response.json()

        pipelines.extend(data.get("items", []))
        print(f"Loaded {len(pipelines)} pipelines", flush=True)

        pageToken = data.get("next_page_token")
        if not pageToken:
            break

        params["page-token"] = pageToken

    return pipelines


def getPipelineWorkflows(pipeline_id):
    url = f"https://circleci.com/api/v2/pipeline/{pipeline_id}/workflow"

    response = requests.get(
        url,
        headers=headers,
        timeout=10
    )

    if response.status_code == 404:
        print(f"Pipeline {pipeline_id}: workflow not found, skipping")
        return []

    response.raise_for_status()

    return response.json().get("items", [])


def getPiplineJob(workflow_id):
    url = f"https://circleci.com/api/v2/workflow/{workflow_id}/job"
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()
    return data.get("items", [])

def get_job_details(job_number):

    url = (
        "https://circleci.com/api/v1.1/"
        f"project/github/EmptyHead5/FEVR/{job_number}"
    )

    response = requests.get(
        url,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    return response.json()



def main():
    pipelines = getAllPipelines()
    filtedPipelines = []

    for pipeline in pipelines:
        if pipeline.get("number", 0) > 246:
            filtedPipelines.append(pipeline)

    print(f"Total pipelines fetched: {len(filtedPipelines)}", flush=True)  # 552-246=306

    for pipeline in filtedPipelines:
        pipline_id = pipeline.get("id")
        tradingWorkflow = None
        workflows = getPipelineWorkflows(pipline_id)

        for workflow in workflows:
            if workflow.get("name") == "hourly-paper-trading":
                tradingWorkflow = workflow
                break

        if tradingWorkflow is None:
            continue

        jobs = getPiplineJob(tradingWorkflow["id"])
        tradingJob = None

        for job in jobs:
            if job.get("name") == "hourly-paper-trading":
                tradingJob = job
                break

        if tradingJob is None:
            continue

        jobNumber = tradingJob["job_number"]
        print(f"Job #{jobNumber}", flush=True)
        jobDetail = get_job_details(jobNumber)

        for step in jobDetail.get("steps", []):
            if step.get("name") == "Run hourly prediction and paper trading":
                for action in step.get("actions", []):
                    output_url = action.get("output_url")

                    if output_url:
                        response = requests.get(output_url, timeout=10)
                        response.raise_for_status()
                        logs = response.json()

                        message = "".join(
                            log.get("message", "") for log in logs
                        )

                        if "Market is open: True" in message:
                            dataTime = probability = threshold = None
                            for line in message.splitlines():
                                if line.startswith("Latest completed data time:"):
                                    dataTime = line.split(": ", 1)[1]
                                elif line.startswith("Up probability:"):
                                    probability = line.split(": ", 1)[1]
                                elif line.startswith("Threshold:"):
                                    threshold = line.split(": ", 1)[1]
                            if dataTime and probability and threshold:
                                print(f"Pipeline #{pipeline['number']} → Job #{jobNumber}", flush=True)
                                results.append({
                                    "job_number": jobNumber,
                                    "data_time": dataTime,
                                    "probability": probability,
                                    "threshold": threshold
                                })

    pd.DataFrame(results,columns=["job_number", "data_time", "probability", "threshold"]).to_csv(csv_file, index=False)

    history = pd.read_csv(csv_file)


    history["data_time"] = pd.to_datetime(history["data_time"], utc=True)
    recent = history.sort_values("data_time", ascending=False).head(48)

    if len(recent) < 48:
        raise ValueError("Not enough historical records (less than 48)")

    probabilities = recent["probability"].str.rstrip("%").astype(float) / 100

    sell_threshold = float(probabilities.quantile(0.25))

    print("Start time:", recent["data_time"].min())
    print("End time:", recent["data_time"].max())
    print(f"Sell threshold: {sell_threshold:.2%}")

    return sell_threshold

if __name__ == "__main__":
    main()
