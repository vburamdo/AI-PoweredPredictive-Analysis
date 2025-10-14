import React from "react";

export default function AnomalyPanel({ anomalies = [] }) {
  if (!anomalies.length) return <div className="text-gray-500">No anomalies</div>;

  return (
    <div className="space-y-3">
      {anomalies.map((a, i) => (
        <div key={i} className="p-3 border rounded">
          <div className="flex justify-between">
            <div className="font-semibold">{a.summary || a.title || "Anomaly"}</div>
            <div className="text-xs text-gray-500">{a.created_at || a.timestamp || ""}</div>
          </div>
          <div className="mt-2 text-sm text-gray-700">
            <strong>Severity:</strong> {a.severity_level || "Unknown"}
            <div className="mt-2"><strong>Probable causes:</strong> {Array.isArray(a.probable_causes) ? a.probable_causes.join(", ") : a.probable_causes}</div>
            <div className="mt-2"><strong>Action:</strong> {a.recommended_action}</div>
          </div>
        </div>
      ))}
    </div>
  );
}
