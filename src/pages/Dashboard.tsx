// @ts-nocheck
import { useState, useEffect, useRef } from "react";
import ModelActivity from "../components/ModelActivity";
import TradingChart from "../components/TradingChart";

const PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD"];

const MOCK_PREDICTIONS = {
  "EUR/USD": {
    price: 1.08432,
    change: +0.0021,
    changePct: +0.19,
    horizons: {
      "1H":  { direction: "bullish", probability: 0.71, expectedMove: +0.18, expectedPips: 19 },
      "4H":  { direction: "bullish", probability: 0.64, expectedMove: +0.41, expectedPips: 44 },
      "1D":  { direction: "bearish", probability: 0.58, expectedMove: -0.67, expectedPips: -72 },
      "7D":  { direction: "bullish", probability: 0.53, expectedMove: +1.12, expectedPips: 121 },
    },
    narrative: "Price is compressing against a key resistance zone with momentum aligning across shorter timeframes. Structure favors continuation, though uncertainty remains elevated near the daily level.",
    confidence: "moderate",
    regime: "trending",
    volatility: "low",
    updated: "2 min ago"
  },
  "GBP/USD": {
    price: 1.26841,
    change: -0.0034,
    changePct: -0.27,
    horizons: {
      "1H":  { direction: "bearish", probability: 0.68, expectedMove: -0.22, expectedPips: -28 },
      "4H":  { direction: "bearish", probability: 0.72, expectedMove: -0.55, expectedPips: -70 },
      "1D":  { direction: "bearish", probability: 0.61, expectedMove: -0.89, expectedPips: -113 },
      "7D":  { direction: "neutral", probability: 0.51, expectedMove: -0.31, expectedPips: -39 },
    },
    narrative: "Bearish momentum is building across multiple timeframes with price failing to reclaim recent highs. Rejection wicks suggest sustained selling pressure at current levels.",
    confidence: "high",
    regime: "trending",
    volatility: "moderate",
    updated: "1 min ago"
  },
  "USD/JPY": {
    price: 149.823,
    change: +0.412,
    changePct: +0.28,
    horizons: {
      "1H":  { direction: "bullish", probability: 0.62, expectedMove: +0.14, expectedPips: 21 },
      "4H":  { direction: "neutral", probability: 0.54, expectedMove: +0.08, expectedPips: 12 },
      "1D":  { direction: "bullish", probability: 0.67, expectedMove: +0.72, expectedPips: 108 },
      "7D":  { direction: "bullish", probability: 0.74, expectedMove: +1.44, expectedPips: 216 },
    },
    narrative: "Strong structural support holding across timeframes with momentum confirming upward bias. The weekly picture shows clear directional intent with volatility remaining contained.",
    confidence: "high",
    regime: "trending",
    volatility: "low",
    updated: "3 min ago"
  },
  "USD/CHF": {
    price: 0.90124,
    change: -0.0008,
    changePct: -0.09,
    horizons: {
      "1H":  { direction: "neutral", probability: 0.51, expectedMove: -0.04, expectedPips: -4 },
      "4H":  { direction: "bearish", probability: 0.59, expectedMove: -0.28, expectedPips: -25 },
      "1D":  { direction: "bearish", probability: 0.63, expectedMove: -0.51, expectedPips: -46 },
      "7D":  { direction: "neutral", probability: 0.52, expectedMove: +0.18, expectedPips: 16 },
    },
    narrative: "Mixed signals across timeframes indicate a consolidating market. Short-term indecision is dominant with no clear structural bias establishing itself at current price levels.",
    confidence: "low",
    regime: "ranging",
    volatility: "low",
    updated: "5 min ago"
  },
  "AUD/USD": {
    price: 0.65234,
    change: +0.0018,
    changePct: +0.28,
    horizons: {
      "1H":  { direction: "bullish", probability: 0.66, expectedMove: +0.21, expectedPips: 14 },
      "4H":  { direction: "bullish", probability: 0.69, expectedMove: +0.48, expectedPips: 31 },
      "1D":  { direction: "bullish", probability: 0.57, expectedMove: +0.74, expectedPips: 48 },
      "7D":  { direction: "bearish", probability: 0.55, expectedMove: -0.92, expectedPips: -60 },
    },
    narrative: "Short to medium-term bullish pressure is present with price reclaiming structure. However, the weekly outlook shows potential exhaustion, suggesting caution on extended holds.",
    confidence: "moderate",
    regime: "ranging",
    volatility: "moderate",
    updated: "2 min ago"
  },
  "USD/CAD": {
    price: 1.36512,
    change: -0.0022,
    changePct: -0.16,
    horizons: {
      "1H":  { direction: "bearish", probability: 0.60, expectedMove: -0.15, expectedPips: -20 },
      "4H":  { direction: "neutral", probability: 0.53, expectedMove: +0.07, expectedPips: 10 },
      "1D":  { direction: "bearish", probability: 0.65, expectedMove: -0.58, expectedPips: -80 },
      "7D":  { direction: "bearish", probability: 0.61, expectedMove: -1.02, expectedPips: -139 },
    },
    narrative: "Bearish structure is dominant across daily and weekly timeframes. Short-term noise is present but the prevailing directional bias remains to the downside at key distribution zones.",
    confidence: "moderate",
    regime: "trending",
    volatility: "moderate",
    updated: "4 min ago"
  },
  "NZD/USD": {
    price: 0.60891,
    change: +0.0011,
    changePct: +0.18,
    horizons: {
      "1H":  { direction: "neutral", probability: 0.52, expectedMove: +0.06, expectedPips: 4 },
      "4H":  { direction: "bullish", probability: 0.61, expectedMove: +0.33, expectedPips: 20 },
      "1D":  { direction: "neutral", probability: 0.54, expectedMove: -0.12, expectedPips: -7 },
      "7D":  { direction: "bullish", probability: 0.58, expectedMove: +0.88, expectedPips: 54 },
    },
    narrative: "Muted short-term price action with structure attempting to form higher lows. Medium-term bias leans bullish but lacks the conviction seen in stronger trending pairs currently.",
    confidence: "low",
    regime: "ranging",
    volatility: "low",
    updated: "6 min ago"
  }
};


function ProbabilityBar({ value, direction }) {
  const pct = Math.round(value * 100);
  const isUp = direction === "bullish";
  const isNeutral = direction === "neutral";
  const color = isNeutral ? "#a0a8b8" : isUp ? "#4fc3f7" : "#ff4757";

  return (
    <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
      <div style={{
        flex: 1, height: "3px", background: "rgba(255,255,255,0.07)",
        borderRadius: "2px", overflow: "hidden"
      }}>
        <div style={{
          width: `${pct}%`, height: "100%",
          background: color,
          borderRadius: "2px",
          transition: "width 0.6s ease",
          boxShadow: `0 0 6px ${color}60`
        }} />
      </div>
      <span style={{
        fontSize: "13px", fontFamily: "'DM Mono', monospace",
        color, fontWeight: "600", minWidth: "34px", textAlign: "right"
      }}>{pct}%</span>
    </div>
  );
}

export default function LumenYDashboard() {
  const [selectedPair, setSelectedPair] = useState("EUR/USD");
  const [selectedHorizon, setSelectedHorizon] = useState("1D");
  const [chatOpen, setChatOpen] = useState(false);
  const [chatMessages, setChatMessages] = useState([
    { role: "assistant", text: "Ask me anything about this pair — market structure, what's driving the current outlook, or how to interpret the probability scores." }
  ]);
  const [chatInput, setChatInput] = useState("");
  const [animKey, setAnimKey] = useState(0);

  const handlePairSelect = (pair) => {
    setSelectedPair(pair);
    setAnimKey(k => k + 1);
  };

  const pred = MOCK_PREDICTIONS[selectedPair];
  const horizon = pred?.horizons[selectedHorizon];
  const isUp = horizon?.direction === "bullish";
  const isNeutral = horizon?.direction === "neutral";
  const directionColor = isNeutral ? "#a0a8b8" : isUp ? "#4fc3f7" : "#ff4757";
  const directionLabel = isNeutral ? "NEUTRAL" : isUp ? "BULLISH" : "BEARISH";

  const sendMessage = () => {
    if (!chatInput.trim()) return;
    const userMsg = { role: "user", text: chatInput };
    const responses = [
      `For ${selectedPair}, the current ${selectedHorizon} outlook reflects ${pred.narrative.toLowerCase()}`,
      `The ${Math.round(horizon?.probability * 100)}% probability is derived from confluence across multiple timeframes. Higher confidence signals appear when shorter and longer timeframe structures agree.`,
      `Volatility is currently ${pred.volatility} for this pair, which means the expected move range is tighter than average. This tends to increase directional reliability.`,
      `The market regime for ${selectedPair} is ${pred.regime}. In ${pred.regime} conditions, momentum-based signals carry more weight in the model's assessment.`
    ];
    const reply = { role: "assistant", text: responses[Math.floor(Math.random() * responses.length)] };
    setChatMessages(m => [...m, userMsg, reply]);
    setChatInput("");
  };

  const confidenceColors = { high: "#4fc3f7", moderate: "#f4a14a", low: "#a0a8b8" };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;500;600;700;800&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #060a10; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }
        .pair-item:hover { background: rgba(255,255,255,0.05) !important; }
        .pair-item.active { background: rgba(255,255,255,0.08) !important; }
        .horizon-btn:hover { background: rgba(255,255,255,0.08) !important; }
        .horizon-btn.active { background: rgba(255,255,255,0.1) !important; border-color: rgba(255,255,255,0.2) !important; }
        .chat-input:focus { outline: none; border-color: rgba(255,255,255,0.15) !important; }
        .send-btn:hover { background: rgba(255,255,255,0.15) !important; }
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulseGlow {
          0%, 100% { opacity: 0.6; }
          50% { opacity: 1; }
        }
        .live-dot { animation: pulseGlow 2s ease-in-out infinite; }
        .fade-up { animation: fadeUp 0.4s ease forwards; }
      `}</style>

      <div style={{
        display: "flex", height: "100vh", background: "#060a10",
        fontFamily: "'Syne', sans-serif", color: "#e8eaf0", overflow: "hidden"
      }}>

        {/* Sidebar */}
        <div style={{
          width: "220px", flexShrink: 0,
          background: "rgba(255,255,255,0.02)",
          borderRight: "1px solid rgba(255,255,255,0.05)",
          display: "flex", flexDirection: "column"
        }}>
          {/* Logo */}
          <div style={{
            padding: "24px 20px 20px",
            borderBottom: "1px solid rgba(255,255,255,0.05)"
          }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
              <img src="/logo-transparent.png" style={{ height: "32px", width: "auto" }} alt="LumenY" />
            </div>
          </div>

          {/* Pairs list */}
          <div style={{ flex: 1, overflowY: "auto", padding: "12px 8px" }}>
            <div style={{
              fontSize: "9px", fontFamily: "'DM Mono', monospace",
              color: "rgba(255,255,255,0.25)", letterSpacing: "0.12em",
              padding: "4px 12px 8px", textTransform: "uppercase"
            }}>FX Majors</div>

            {PAIRS.map(pair => {
              const p = MOCK_PREDICTIONS[pair];
              const d1 = p.horizons["1D"];
              const isActive = pair === selectedPair;
              const pColor = d1.direction === "bullish" ? "#4fc3f7" : d1.direction === "bearish" ? "#ff4757" : "#a0a8b8";

              return (
                <div
                  key={pair}
                  className={`pair-item ${isActive ? "active" : ""}`}
                  onClick={() => handlePairSelect(pair)}
                  style={{
                    padding: "10px 12px",
                    borderRadius: "8px",
                    cursor: "pointer",
                    marginBottom: "2px",
                    background: isActive ? "rgba(255,255,255,0.08)" : "transparent",
                    transition: "background 0.15s",
                    borderLeft: isActive ? `2px solid ${pColor}` : "2px solid transparent"
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontSize: "12px", fontWeight: "600", letterSpacing: "0.02em" }}>{pair}</span>
                    <span style={{
                      fontSize: "9px", fontFamily: "'DM Mono', monospace",
                      color: pColor, fontWeight: "500"
                    }}>{Math.round(d1.probability * 100)}%</span>
                  </div>
                  <div style={{
                    fontSize: "10px", fontFamily: "'DM Mono', monospace",
                    color: "rgba(255,255,255,0.35)", marginTop: "2px"
                  }}>{p.price.toFixed(pair.includes("JPY") ? 3 : 5)}</div>
                </div>
              );
            })}
          </div>

          {/* Bottom status */}
          <div style={{
            padding: "16px 20px",
            borderTop: "1px solid rgba(255,255,255,0.05)"
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
              <div className="live-dot" style={{
                width: "6px", height: "6px", borderRadius: "50%",
                background: "#4fc3f7"
              }} />
              <span style={{
                fontSize: "10px", fontFamily: "'DM Mono', monospace",
                color: "rgba(255,255,255,0.3)"
              }}>Live · Updated {pred.updated}</span>
            </div>
          </div>
        </div>

        {/* Main content */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>

          {/* Header */}
          <div style={{
            padding: "20px 28px 16px",
            borderBottom: "1px solid rgba(255,255,255,0.05)",
            display: "flex", alignItems: "center", justifyContent: "space-between",
            flexShrink: 0
          }}>
            <div style={{ display: "flex", alignItems: "baseline", gap: "16px" }}>
              <h1 style={{ fontSize: "22px", fontWeight: "700", letterSpacing: "-0.01em" }}>
                {selectedPair}
              </h1>
              <span style={{
                fontSize: "18px", fontFamily: "'DM Mono', monospace",
                color: "rgba(255,255,255,0.6)", fontWeight: "300"
              }}>{pred.price.toFixed(selectedPair.includes("JPY") ? 3 : 5)}</span>
              <span style={{
                fontSize: "12px", fontFamily: "'DM Mono', monospace",
                color: pred.changePct >= 0 ? "#4fc3f7" : "#ff4757"
              }}>
                {pred.changePct >= 0 ? "+" : ""}{pred.changePct.toFixed(2)}%
              </span>
            </div>

            {/* Horizon selector */}
            <div style={{ display: "flex", gap: "6px" }}>
              {["1H", "4H", "1D", "7D"].map(h => (
                <button
                  key={h}
                  className={`horizon-btn ${selectedHorizon === h ? "active" : ""}`}
                  onClick={() => setSelectedHorizon(h)}
                  style={{
                    padding: "6px 14px",
                    background: selectedHorizon === h ? "rgba(255,255,255,0.1)" : "transparent",
                    border: `1px solid ${selectedHorizon === h ? "rgba(255,255,255,0.2)" : "rgba(255,255,255,0.08)"}`,
                    borderRadius: "6px",
                    color: selectedHorizon === h ? "#fff" : "rgba(255,255,255,0.4)",
                    fontSize: "11px", fontFamily: "'DM Mono', monospace",
                    cursor: "pointer", fontWeight: "500",
                    letterSpacing: "0.05em", transition: "all 0.15s"
                  }}
                >{h}</button>
              ))}
            </div>

            {/* Chat toggle */}
            <button
              onClick={() => setChatOpen(o => !o)}
              style={{
                padding: "8px 16px",
                background: chatOpen ? "rgba(79,195,247,0.1)" : "rgba(255,255,255,0.05)",
                border: `1px solid ${chatOpen ? "rgba(79,195,247,0.3)" : "rgba(255,255,255,0.08)"}`,
                borderRadius: "8px",
                color: chatOpen ? "#4fc3f7" : "rgba(255,255,255,0.5)",
                fontSize: "12px", fontFamily: "'Syne', sans-serif",
                cursor: "pointer", fontWeight: "600",
                transition: "all 0.2s", display: "flex", alignItems: "center", gap: "6px"
              }}
            >
              <span style={{ fontSize: "14px" }}>◎</span>
              Ask LumenY
            </button>
          </div>

          {/* Chart + Insights row */}
          <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>

            {/* Chart area */}
            <div style={{
              flex: 1, display: "flex", flexDirection: "column",
              padding: "20px 24px", overflow: "hidden"
            }}>
              {/* Chart */}
              <div style={{
                flex: 1,
                background: "#080c14",
                borderRadius: "12px",
                border: "1px solid rgba(255,255,255,0.06)",
                overflow: "hidden",
                position: "relative",
                minHeight: 0
              }}>
                <TradingChart
                  key={selectedPair}
                  pair={selectedPair}
                  prediction={pred}
                  horizon={selectedHorizon}
                />
              </div>

              {/* All horizons mini row */}
              <div style={{
                display: "flex", gap: "10px", marginTop: "12px", flexShrink: 0
              }}>
                {Object.entries(pred.horizons).map(([h, data]) => {
                  const hUp = data.direction === "bullish";
                  const hNeutral = data.direction === "neutral";
                  const hColor = hNeutral ? "#a0a8b8" : hUp ? "#4fc3f7" : "#ff4757";
                  const isSelected = h === selectedHorizon;

                  return (
                    <div
                      key={h}
                      onClick={() => setSelectedHorizon(h)}
                      style={{
                        flex: 1, padding: "12px 14px",
                        background: isSelected ? "rgba(255,255,255,0.06)" : "rgba(255,255,255,0.02)",
                        border: `1px solid ${isSelected ? "rgba(255,255,255,0.1)" : "rgba(255,255,255,0.04)"}`,
                        borderRadius: "10px", cursor: "pointer",
                        transition: "all 0.2s"
                      }}
                    >
                      <div style={{
                        fontSize: "9px", fontFamily: "'DM Mono', monospace",
                        color: "rgba(255,255,255,0.3)", letterSpacing: "0.1em",
                        marginBottom: "6px"
                      }}>{h}</div>
                      <div style={{
                        fontSize: "18px", fontWeight: "700",
                        color: hColor, fontFamily: "'DM Mono', monospace"
                      }}>{Math.round(data.probability * 100)}%</div>
                      <div style={{
                        fontSize: "10px", color: hColor,
                        marginTop: "2px", fontFamily: "'DM Mono', monospace",
                        opacity: 0.7
                      }}>
                        {data.direction === "neutral" ? "—" : hUp ? "▲" : "▼"} {Math.abs(data.expectedMove).toFixed(2)}%
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Model Activity */}
              <ModelActivity pair={selectedPair} />
            </div>

            {/* Right panel */}
            <div style={{
              width: chatOpen ? "300px" : "280px",
              flexShrink: 0,
              borderLeft: "1px solid rgba(255,255,255,0.05)",
              display: "flex", flexDirection: "column",
              overflow: "hidden", transition: "width 0.3s ease"
            }}>

              {chatOpen ? (
                /* Chat panel */
                <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
                  <div style={{
                    padding: "16px 20px 12px",
                    borderBottom: "1px solid rgba(255,255,255,0.05)"
                  }}>
                    <div style={{ fontSize: "13px", fontWeight: "600", color: "#4fc3f7" }}>Ask LumenY</div>
                    <div style={{
                      fontSize: "10px", fontFamily: "'DM Mono', monospace",
                      color: "rgba(255,255,255,0.25)", marginTop: "2px"
                    }}>{selectedPair} context active</div>
                  </div>

                  <div style={{
                    flex: 1, overflowY: "auto",
                    padding: "16px 16px 8px", display: "flex",
                    flexDirection: "column", gap: "10px"
                  }}>
                    {chatMessages.map((msg, i) => (
                      <div key={i} style={{
                        display: "flex",
                        justifyContent: msg.role === "user" ? "flex-end" : "flex-start"
                      }}>
                        <div style={{
                          maxWidth: "85%",
                          padding: "10px 12px",
                          background: msg.role === "user"
                            ? "rgba(79,195,247,0.12)"
                            : "rgba(255,255,255,0.04)",
                          border: `1px solid ${msg.role === "user" ? "rgba(79,195,247,0.2)" : "rgba(255,255,255,0.06)"}`,
                          borderRadius: msg.role === "user" ? "12px 12px 2px 12px" : "12px 12px 12px 2px",
                          fontSize: "12px", lineHeight: "1.5",
                          color: msg.role === "user" ? "#e8eaf0" : "rgba(255,255,255,0.7)"
                        }}>{msg.text}</div>
                      </div>
                    ))}
                  </div>

                  <div style={{ padding: "12px 16px", borderTop: "1px solid rgba(255,255,255,0.05)" }}>
                    <div style={{ display: "flex", gap: "8px" }}>
                      <input
                        className="chat-input"
                        value={chatInput}
                        onChange={e => setChatInput(e.target.value)}
                        onKeyDown={e => e.key === "Enter" && sendMessage()}
                        placeholder="Ask about this pair..."
                        style={{
                          flex: 1, background: "rgba(255,255,255,0.04)",
                          border: "1px solid rgba(255,255,255,0.08)",
                          borderRadius: "8px", padding: "8px 12px",
                          color: "#e8eaf0", fontSize: "12px",
                          fontFamily: "'Syne', sans-serif",
                          transition: "border-color 0.2s"
                        }}
                      />
                      <button
                        className="send-btn"
                        onClick={sendMessage}
                        style={{
                          width: "34px", height: "34px",
                          background: "rgba(255,255,255,0.07)",
                          border: "1px solid rgba(255,255,255,0.1)",
                          borderRadius: "8px", color: "#4fc3f7",
                          cursor: "pointer", fontSize: "14px",
                          transition: "background 0.15s",
                          display: "flex", alignItems: "center", justifyContent: "center"
                        }}
                      >→</button>
                    </div>
                  </div>
                </div>

              ) : (
                /* Insights panel */
                <div style={{ flex: 1, overflowY: "auto", padding: "20px 20px" }}>

                  {/* Main probability card */}
                  <div key={`${selectedPair}-${selectedHorizon}-${animKey}`} className="fade-up" style={{
                    padding: "20px",
                    background: `linear-gradient(135deg, ${directionColor}0d, rgba(255,255,255,0.02))`,
                    border: `1px solid ${directionColor}30`,
                    borderRadius: "12px",
                    marginBottom: "16px"
                  }}>
                    <div style={{
                      fontSize: "9px", fontFamily: "'DM Mono', monospace",
                      color: "rgba(255,255,255,0.3)", letterSpacing: "0.12em",
                      marginBottom: "12px", textTransform: "uppercase"
                    }}>{selectedHorizon} Outlook · {selectedPair}</div>

                    <div style={{
                      fontSize: "48px", fontWeight: "800",
                      color: directionColor, lineHeight: 1,
                      fontFamily: "'DM Mono', monospace",
                      marginBottom: "4px"
                    }}>{Math.round(horizon?.probability * 100)}%</div>

                    <div style={{
                      fontSize: "11px", letterSpacing: "0.15em",
                      color: directionColor, fontWeight: "600",
                      marginBottom: "14px", opacity: 0.8
                    }}>{directionLabel}</div>

                    <div style={{
                      padding: "10px 12px",
                      background: "rgba(255,255,255,0.03)",
                      borderRadius: "8px",
                      border: "1px solid rgba(255,255,255,0.06)"
                    }}>
                      <div style={{
                        fontSize: "9px", fontFamily: "'DM Mono', monospace",
                        color: "rgba(255,255,255,0.25)", marginBottom: "4px",
                        letterSpacing: "0.1em"
                      }}>EXPECTED MOVE</div>
                      <div style={{
                        fontSize: "16px", fontFamily: "'DM Mono', monospace",
                        color: directionColor, fontWeight: "500"
                      }}>
                        {horizon?.expectedMove >= 0 ? "+" : ""}{horizon?.expectedMove.toFixed(2)}%
                        <span style={{
                          fontSize: "11px", color: "rgba(255,255,255,0.3)",
                          marginLeft: "8px"
                        }}>{horizon?.expectedPips > 0 ? "+" : ""}{horizon?.expectedPips} pips</span>
                      </div>
                    </div>
                  </div>

                  {/* Narrative */}
                  <div style={{
                    padding: "14px 16px",
                    background: "rgba(255,255,255,0.02)",
                    border: "1px solid rgba(255,255,255,0.05)",
                    borderRadius: "10px",
                    marginBottom: "16px"
                  }}>
                    <div style={{
                      fontSize: "9px", fontFamily: "'DM Mono', monospace",
                      color: "rgba(255,255,255,0.25)", letterSpacing: "0.12em",
                      marginBottom: "8px", textTransform: "uppercase"
                    }}>Market Reading</div>
                    <p style={{
                      fontSize: "12px", lineHeight: "1.65",
                      color: "rgba(255,255,255,0.55)", fontWeight: "400"
                    }}>{pred.narrative}</p>
                  </div>

                  {/* All horizons probabilities */}
                  <div style={{
                    padding: "14px 16px",
                    background: "rgba(255,255,255,0.02)",
                    border: "1px solid rgba(255,255,255,0.05)",
                    borderRadius: "10px",
                    marginBottom: "16px"
                  }}>
                    <div style={{
                      fontSize: "9px", fontFamily: "'DM Mono', monospace",
                      color: "rgba(255,255,255,0.25)", letterSpacing: "0.12em",
                      marginBottom: "12px", textTransform: "uppercase"
                    }}>All Horizons</div>

                    {Object.entries(pred.horizons).map(([h, data]) => {
                      const hUp = data.direction === "bullish";
                      const hNeutral = data.direction === "neutral";
                      const hColor = hNeutral ? "#a0a8b8" : hUp ? "#4fc3f7" : "#ff4757";
                      return (
                        <div key={h} style={{ marginBottom: "10px" }} onClick={() => setSelectedHorizon(h)}>
                          <div style={{
                            display: "flex", justifyContent: "space-between",
                            marginBottom: "5px", cursor: "pointer"
                          }}>
                            <span style={{
                              fontSize: "10px", fontFamily: "'DM Mono', monospace",
                              color: h === selectedHorizon ? "#fff" : "rgba(255,255,255,0.35)",
                              letterSpacing: "0.08em"
                            }}>{h}</span>
                            <span style={{
                              fontSize: "10px", fontFamily: "'DM Mono', monospace",
                              color: hColor, opacity: 0.7
                            }}>{hNeutral ? "—" : hUp ? "▲" : "▼"} {Math.abs(data.expectedMove).toFixed(2)}%</span>
                          </div>
                          <ProbabilityBar value={data.probability} direction={data.direction} />
                        </div>
                      );
                    })}
                  </div>

                  {/* Market conditions */}
                  <div style={{
                    display: "grid", gridTemplateColumns: "1fr 1fr",
                    gap: "8px"
                  }}>
                    {[
                      { label: "Regime", value: pred.regime, icon: "◈" },
                      { label: "Volatility", value: pred.volatility, icon: "◎" },
                      { label: "Confidence", value: pred.confidence, icon: "◇", color: confidenceColors[pred.confidence] },
                      { label: "Updated", value: pred.updated, icon: "◉" }
                    ].map(item => (
                      <div key={item.label} style={{
                        padding: "10px 12px",
                        background: "rgba(255,255,255,0.02)",
                        border: "1px solid rgba(255,255,255,0.05)",
                        borderRadius: "8px"
                      }}>
                        <div style={{
                          fontSize: "8px", fontFamily: "'DM Mono', monospace",
                          color: "rgba(255,255,255,0.2)", letterSpacing: "0.1em",
                          marginBottom: "4px", textTransform: "uppercase"
                        }}>{item.label}</div>
                        <div style={{
                          fontSize: "11px", fontFamily: "'DM Mono', monospace",
                          color: item.color || "rgba(255,255,255,0.6)",
                          fontWeight: "500", textTransform: "capitalize"
                        }}>{item.value}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}