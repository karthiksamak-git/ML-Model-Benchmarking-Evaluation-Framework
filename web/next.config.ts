import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // The dashboard is fully static: all numbers are pre-exported JSON from
  // local MLflow runs. No Python, MLflow or training runs on Vercel.
  output: "export",
  images: { unoptimized: true },
  // Cloud preview origins are supplied per process; never a blanket allowlist.
  allowedDevOrigins: (process.env.NEXT_ALLOWED_DEV_ORIGINS ?? "")
    .split(",")
    .filter(Boolean),
};

export default nextConfig;
