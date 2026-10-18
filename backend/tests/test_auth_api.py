"""Match-day PIN roles: open mode without PINs; operator vs umpire permissions when configured."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("DRS_STORAGE_DIR", str(tmp_path_factory.mktemp("storage")))
    mp.setenv("DRS_OPERATOR_PIN", "op-1234")
    mp.setenv("DRS_UMPIRE_PIN", "ump-9876")
    from backend.app.core.config import get_settings

    get_settings.cache_clear()
    from backend.app.main import app

    with TestClient(app) as c:
        yield c
    mp.undo()
    get_settings.cache_clear()


OP, UMP = {"X-DRS-PIN": "op-1234"}, {"X-DRS-PIN": "ump-9876"}


def test_roles(client):
    assert client.get("/api/auth/status").json() == {"enabled": True, "role": "viewer"}
    assert client.get("/api/auth/status", headers=OP).json()["role"] == "operator"
    assert client.get("/api/auth/status", headers=UMP).json()["role"] == "umpire"
    assert client.get("/api/matches").status_code == 200  # reading stays open

    pid = client.get("/api/protocols").json()[0]["id"]
    body = {"name": "m", "team_a": "A", "team_b": "B", "protocol_id": pid}
    assert client.post("/api/matches", json=body).status_code == 401
    assert client.post("/api/matches", json=body, headers={"X-DRS-PIN": "wrong"}).status_code == 401
    m = client.post("/api/matches", json=body, headers=OP)
    assert m.status_code == 201
    rv = client.post(f"/api/matches/{m.json()['id']}/reviews", headers=OP,
                     json={"requested_by": "A", "decision_type": "lbw", "on_field": "OUT"}).json()
    dec = {"final_decision": "OUT", "outcome": "UPHELD", "decided_by": "TV"}
    assert client.post(f"/api/reviews/{rv['id']}/decide", json=dec, headers=OP).status_code == 403
    assert client.post(f"/api/reviews/{rv['id']}/decide", json=dec, headers=UMP).status_code == 200
