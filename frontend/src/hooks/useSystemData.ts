import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, getStats, getSystemStatus } from "@/lib/api";
import type { Stats, SystemStatus } from "@/types/api";

/**
 * Loads the live analytics card data and component health report, and keeps
 * them fresh with a light polling loop so a second browser tab (or the
 * ingestion pipeline running elsewhere) shows up automatically.
 */
export function useSystemData(pollIntervalMs = 15_000) {
  const [stats, setStats] = useState<Stats | null>(null);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(true);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const refresh = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);
    try {
      // Both endpoints are independent - fetch them concurrently.
      const [nextStats, nextStatus] = await Promise.all([getStats(), getSystemStatus()]);
      if (!mounted.current) return;
      setStats(nextStats);
      setStatus(nextStatus);
      setError(null);
    } catch (cause) {
      if (!mounted.current) return;
      setError(
        cause instanceof ApiError
          ? cause
          : new ApiError(cause instanceof Error ? cause.message : String(cause)),
      );
    } finally {
      if (mounted.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (!pollIntervalMs) return undefined;
    const id = window.setInterval(() => {
      if (document.visibilityState === "visible") void refresh(true);
    }, pollIntervalMs);
    return () => window.clearInterval(id);
  }, [pollIntervalMs, refresh]);

  return { stats, status, error, loading, refresh };
}
