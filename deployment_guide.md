# How to Deploy Flex-ID

The easiest way to deploy this project is using **Docker**. This allows the Node.js Backend and Python ML Engine to run in the same environment.

## Option A: Deploy to Render (Free/Cheap)
Render is a cloud platform that supports Docker natively.

1.  **Push to GitHub**: Ensure your latest code is pushed to your GitHub repository.
2.  **Create Account**: Go to [render.com](https://render.com) and create an account.
3.  **New Web Service**:
    *   Click "New +" -> "Web Service".
    *   Connect your GitHub repository.
4.  **Configuration**:
    *   **Runtime**: Select **Docker**.
    *   **Region**: Nearest to you (e.g., Singapore, Oregon).
    *   **Branch**: `main` (or your active branch).
    *   **Plan**: Free (Note: The Free plan might be slow for Training, a paid plan is better for ML workloads).
5.  **Environment Variables**:
    *   Add any secrets if you have them (e.g., if you add DB credentials later).
6.  **Deploy**: Click "Create Web Service". Render will build the Docker image (this takes ~5-10 mins) and start it.

## Option B: Run Locally with Docker
If you have Docker Desktop installed:

1.  **Build the Image**:
    ```bash
    docker build -t flex-id .
    ```
2.  **Run the Container**:
    ```bash
    docker run -p 5000:5000 flex-id
    ```
3.  **Access App**: Open `http://localhost:5000` in your browser.

## Notes
- **Data Persistence**: On unpaid container hosting (like Render Free), the filesystem is ephemeral. If the container restarts, `results/` and uploaded datasets might be lost. To fix this, you would need to mount a **Persistent Disk** (Render Paid Feature) or use cloud storage (S3).
- **Performance**: ML training (TensorFlow) is heavy. A basic free tier might run out of memory (RAM). If clients crash with OOM (Out Of Memory), try reducing `--batch_size` in the manual controls page (e.g., set to 32 or 64).
