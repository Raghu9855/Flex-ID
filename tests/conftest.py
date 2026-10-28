import pytest
from typing import Generator, Dict, Any, List
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.base import Base
from app.db.session import get_db
from app.db.seeder import seed_database
from app.schemas.prediction import CANONICAL_FEATURES

# In-memory SQLite for fast, isolated testing
TEST_DATABASE_URL = "sqlite:///:memory:"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    Base.metadata.create_all(bind=test_engine)
    db = TestingSessionLocal()
    seed_database(db)
    db.close()
    yield
    Base.metadata.drop_all(bind=test_engine)

@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    connection = test_engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture
def client(db_session: Session) -> Generator[TestClient, None, None]:
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

@pytest.fixture
def sample_benign_features() -> Dict[str, float]:
    """Sample benign web flow features."""
    return {
        "Dst Port": 80.0,
        "Protocol": 6.0,
        "Flow Duration": 125000.0,
        "Tot Fwd Pkts": 8.0,
        "Tot Bwd Pkts": 7.0,
        "TotLen Fwd Pkts": 450.0,
        "TotLen Bwd Pkts": 1200.0,
        "Fwd Pkt Len Max": 150.0,
        "Fwd Pkt Len Mean": 56.25,
        "Bwd Pkt Len Max": 350.0,
        "Bwd Pkt Len Mean": 171.4,
        "Flow Byts/s": 13200.0,
        "Flow Pkts/s": 120.0,
        "Flow IAT Mean": 8900.0,
        "Flow IAT Max": 24000.0,
        "Fwd IAT Mean": 11000.0,
        "Bwd IAT Mean": 14000.0,
        "Fwd Header Len": 160.0,
        "Bwd Header Len": 140.0,
        "Fwd Pkts/s": 64.0,
        "Bwd Pkts/s": 56.0,
        "Pkt Len Mean": 110.0,
        "Pkt Len Max": 350.0,
        "Pkt Len Var": 12500.0,
        "SYN Flag Cnt": 0.0,
        "RST Flag Cnt": 0.0,
        "ACK Flag Cnt": 1.0,
        "Init Fwd Win Byts": 26883.0,
        "Init Bwd Win Byts": 26883.0,
    }

@pytest.fixture
def sample_feature_list() -> List[float]:
    """Ordered 29-element normalized list."""
    return [0.05] * len(CANONICAL_FEATURES)
