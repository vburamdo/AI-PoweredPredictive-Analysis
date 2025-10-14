import React from "react";

export default function LogTable({ logs = [], loading }) {
  if (loading) return <div>Loading logs…</div>;
  if (!logs.length) return <div className="text-gray-500">No logs</div>;

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left border-b">
            <th className="py-2">Timestamp</th>
            <th>Service</th>
            <th>Level</th>
            <th>Message</th>
          </tr>
        </thead>
        <tbody>
          {logs.map((l, i) => (
            <tr key={i} className="border-b hover:bg-gray-50">
              <td className="py-2">{l.timestamp || l.created_at || "N/A"}</td>
              <td>{l.service_name || "—"}</td>
              <td>
                <span className={`px-2 py-0.5 rounded ${l.log_level === "ERROR" ? "bg-red-100 text-red-700" : l.log_level === "WARN" ? "bg-yellow-100 text-yellow-700" : "bg-green-100 text-green-700"}`}>
                  {l.log_level || "INFO"}
                </span>
              </td>
              <td>{l.message}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
