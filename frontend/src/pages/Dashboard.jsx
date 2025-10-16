import React, { useEffect, useState } from "react";
import { fetchAndStoreLogs } from "../api/api";
import { firestore } from "../firebaseConfig";
import {
    collection,
    query,
    orderBy,
    limit,
    onSnapshot,
} from "firebase/firestore";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import ChatPanel from "../components/ChatPanel";
import "./Dashboard.css";

export default function Dashboard() {
    const [tenantId, setTenantId] = useState("CUST001");
    const [appId, setAppId] = useState("APP01");
    const [logs, setLogs] = useState([]);
    const [anomalies, setAnomalies] = useState([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        if (!tenantId || !appId) return;

        const logsRef = collection(firestore, "tenants", tenantId, "apps", appId, "logs");
        const anomaliesRef = collection(firestore, "tenants", tenantId, "apps", appId, "anomalies");

        const logsQuery = query(logsRef, orderBy("timestamp", "desc"), limit(20));
        const anomaliesQuery = query(anomaliesRef, orderBy("created_at", "desc"), limit(10));

        const unsubLogs = onSnapshot(logsQuery, (snapshot) => {
            setLogs(snapshot.docs.map((doc) => ({ id: doc.id, ...doc.data() })));
        });

        const unsubAnomalies = onSnapshot(anomaliesQuery, (snapshot) => {
            setAnomalies(snapshot.docs.map((doc) => ({ id: doc.id, ...doc.data() })));
        });

        return () => {
            unsubLogs();
            unsubAnomalies();
        };
    }, [tenantId, appId]);

    useEffect(() => {
        const interval = setInterval(async () => {
          try {
            console.log("⏱️ Auto-generating logs for", tenantId, appId);
            await fetchAndStoreLogs(tenantId, appId);
          } catch (err) {
            console.error("Auto-generate failed", err);
          }
        }, 5 * 60 * 1000); // every 5 minutes
      
        return () => clearInterval(interval);
      }, [tenantId, appId]);
      

    useEffect(() => {
        const saved = localStorage.getItem("theme");
        if (saved === "dark") document.body.classList.add("dark-mode");
    }, []);


    const handleGenerate = async () => {
        setLoading(true);
        try {
            await fetchAndStoreLogs(tenantId, appId);
        } catch (err) {
            console.error(err);
            // alert("Failed to generate logs");
        } finally {
            setLoading(false);
        }
    };

    const logVolumeChart = logs.reduce((acc, log) => {
        const time = new Date(log.timestamp || Date.now()).toLocaleTimeString();
        const existing = acc.find((x) => x.time === time);
        if (existing) existing.count++;
        else acc.push({ time, count: 1 });
        return acc;
    }, []);

    return (
        <div className="dashboard-container">
            <header>
                <h1>AI-Powered Predictive-Analysis</h1>

                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                    <input value={tenantId} onChange={(e) => setTenantId(e.target.value)} />
                    <input value={appId} onChange={(e) => setAppId(e.target.value)} />
                    <button onClick={handleGenerate}>
                        {loading ? "Generating..." : "Generate Logs"}
                    </button>

                    <button
                        onClick={() => {
                            const isDark = document.body.classList.toggle("dark-mode");
                            localStorage.setItem("theme", isDark ? "dark" : "light");
                        }}
                        style={{ marginLeft: "8px" }}
                    >
                        🌓
                    </button>
                </div>
            </header>


            <div className="main-grid">
                <div>
                    <div className="card">
                        <h2>Realtime Logs</h2>
                        <table className="logs-table">
                            <thead>
                                <tr>
                                    <th>Timestamp</th>
                                    <th>Level</th>
                                    <th>Service</th>
                                    <th>Message</th>
                                </tr>
                            </thead>
                            <tbody>
                                {loading
                                    ? [...Array(5)].map((_, i) => (
                                        <tr key={i}>
                                            <td colSpan="4">
                                                <div className="skeleton skeleton-line"></div>
                                            </td>
                                        </tr>
                                    ))
                                    : logs.map((log) => (
                                        <tr key={log.id}>
                                            <td>{log.timestamp}</td>
                                            <td>{log.log_level}</td>
                                            <td>{log.service_name}</td>
                                            <td>{log.message}</td>
                                        </tr>
                                    ))}

                            </tbody>
                        </table>
                    </div>

                    <div className="card">
                        <h2>Anomalies</h2>
                        <ul className="anomaly-list">
                            {loading
                                ? [...Array(3)].map((_, i) => (
                                    <li key={i} className="skeleton skeleton-line"></li>
                                ))
                                : anomalies.map((a) => (
                                    <li key={a.id} className="anomaly-item">
                                        <strong>{a.severity_level}</strong>: {a.summary}
                                    </li>
                                ))}

                        </ul>
                    </div>

                    <div className="card chart-container">
                        <h2>Log Volume</h2>
                        <ResponsiveContainer width="100%" height={250}>
                            <LineChart data={logVolumeChart}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="time" />
                                <YAxis />
                                <Tooltip />
                                <Line type="monotone" dataKey="count" stroke="#2563eb" />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div>
                    <div className="card chat-panel">
                        <h2>AI Assistant</h2>
                        <ChatPanel tenantId={tenantId} appId={appId} />
                    </div>
                </div>
            </div>
        </div>
    );
}
