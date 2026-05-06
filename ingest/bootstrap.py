import lakefs
from lakefs.client import Client
from ingest.settings import settings

def run():
    lakefsClient = Client(
        host=settings.lakefs_host,
        username=settings.lakefs_installation_access_key_id,
        password=settings.lakefs_installation_secret_access_key,
    )

    repo = lakefs.Repository("f1-race-data", client=lakefsClient).create(
        storage_namespace="local://f1-race-data",
        exist_ok=True
    )

    branch = repo.branch("main")

    local_files = [
        ("raw/circuits.csv", "data/circuits.csv"),
        ("raw/constructor_results.csv", "data/constructor_results.csv"),
        ("raw/constructor_standings.csv", "data/constructor_standings.csv"),
        ("raw/constructors.csv", "data/constructors.csv"),
        ("raw/driver_standings.csv", "data/driver_standings.csv"),
        ("raw/drivers.csv", "data/drivers.csv"),
        ("raw/lap_times.csv", "data/lap_times.csv"),
        ("raw/pit_stops.csv", "data/pit_stops.csv"),
        ("raw/races.csv", "data/races.csv"),
        ("raw/results.csv", "data/results.csv"),
        ("raw/seasons.csv", "data/seasons.csv"),
        ("raw/sprint_results.csv", "data/sprint_results.csv"),
        ("raw/status.csv", "data/status.csv")
    ]

    try:
        with branch.transact(commit_message="Bootstrap: add raw CSV datasets") as tx:
            for (lakefs_path, local_path) in local_files:
                with open(local_path, mode="rb") as r_f:
                    tx.object(lakefs_path).upload(r_f.read())
                print(f"Staged {lakefs_path}")

        print("Transaction committed to main")

    except Exception as e:
        print(f"Transaction failed and rolled back: {e}")