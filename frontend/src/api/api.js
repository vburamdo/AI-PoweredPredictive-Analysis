import axios from "axios";

const API_BASE = process.env.REACT_APP_API_BASE || "http://127.0.0.1:8000";

export const fetchAndStoreLogs = (tenantId, appId) =>
  axios.post(`${API_BASE}/fetch_and_store_logs?tenant_id=${tenantId}&app_id=${appId}`);

export const storeLogs = (tenantId, appId, rawLogs) =>
  axios.post(`${API_BASE}/store_logs`, { tenant_id: tenantId, app_id: appId, raw_logs: rawLogs });

export const askAI = ({ username, tenant_id, app_id, convo_id, question }) =>
  axios.post(`${API_BASE}/ask_ai`, { username, tenant_id, app_id, convo_id, question });

export const getRecentLogs = async (tenantId, appId, limit = 50) => {
  // You may need a custom endpoint to fetch logs; here we'll assume Firestore query via backend exists
  const res = await axios.get(`${API_BASE}/recent_logs`, { params: { tenant_id: tenantId, app_id: appId, limit } });
  return res.data;
};

export const getAnomalies = async (tenantId, appId, limit = 20) => {
  const res = await axios.get(`${API_BASE}/recent_anomalies`, { params: { tenant_id: tenantId, app_id: appId, limit } });
  return res.data;
};
