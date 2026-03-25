// @ts-nocheck
import { useEffect, useRef, useState } from "react";
import { motion, useInView } from "framer-motion";
import Layout from "../components/Layout";
import ContactModal from "../components/ContactModal";

/* ─── animation helpers ─── */
const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  visible: (i: number = 0) => ({
    opacity: 1,
    y: 0,
    transition: { duration: 0.7, delay: i * 0.12, ease: [0.25, 0.46, 0.45, 0.94] },
  }),
};

function Section({
  children,
  id,
  style,
}: {
  children: React.ReactNode;
  id?: string;
  style?: React.CSSProperties;
}) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-80px" });
  return (
    <motion.section
      ref={ref}
      id={id}
      initial="hidden"
      animate={inView ? "visible" : "hidden"}
      className="section-container"
      style={{ padding: "120px 80px", maxWidth: 1440, margin: "0 auto", ...style }}
    >
      {children}
    </motion.section>
  );
}

/* ─── animated counter ─── */
function Counter({ value, suffix = "", prefix = "", decimals = 0, duration = 2000 }: {
  value: number; suffix?: string; prefix?: string; decimals?: number; duration?: number;
}) {
  const [display, setDisplay] = useState(0);
  const ref = useRef(null);
  const inView = useInView(ref, { once: true });

  useEffect(() => {
    if (!inView) return;
    const start = Date.now();
    const tick = () => {
      const elapsed = Date.now() - start;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplay(eased * value);
      if (progress < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [inView, value, duration]);

  return (
    <span ref={ref}>
      {prefix}{display.toFixed(decimals)}{suffix}
    </span>
  );
}

/* ─── grid background canvas ─── */
function GridBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animId: number;
    let time = 0;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    const draw = () => {
      time += 0.003;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const spacing = 60;
      const cols = Math.ceil(canvas.width / spacing) + 1;
      const rows = Math.ceil(canvas.height / spacing) + 1;

      for (let i = 0; i < cols; i++) {
        for (let j = 0; j < rows; j++) {
          const x = i * spacing;
          const y = j * spacing;
          const dist = Math.sqrt(
            Math.pow(x - canvas.width * 0.5, 2) + Math.pow(y - canvas.height * 0.4, 2)
          );
          const wave = Math.sin(dist * 0.005 - time * 2) * 0.5 + 0.5;
          const alpha = wave * 0.06 * (1 - dist / (canvas.width * 0.7));

          if (alpha > 0.005) {
            ctx.beginPath();
            ctx.arc(x, y, 1, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(79, 195, 247, ${Math.max(0, alpha)})`;
            ctx.fill();
          }
        }
      }

      // Subtle connecting lines
      ctx.strokeStyle = "rgba(79, 195, 247, 0.015)";
      ctx.lineWidth = 0.5;
      for (let i = 0; i < cols; i++) {
        ctx.beginPath();
        ctx.moveTo(i * spacing, 0);
        ctx.lineTo(i * spacing, canvas.height);
        ctx.stroke();
      }
      for (let j = 0; j < rows; j++) {
        ctx.beginPath();
        ctx.moveTo(0, j * spacing);
        ctx.lineTo(canvas.width, j * spacing);
        ctx.stroke();
      }

      animId = requestAnimationFrame(draw);
    };

    draw();
    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
      }}
    />
  );
}

/* ─── live pulse indicator ─── */
function LivePulse() {
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 8,
        fontSize: 11,
        fontFamily: "'DM Mono', monospace",
        color: "#4fc3f7",
        letterSpacing: "0.1em",
        textTransform: "uppercase",
      }}
    >
      <span
        style={{
          width: 6,
          height: 6,
          borderRadius: "50%",
          background: "#4fc3f7",
          animation: "pulse 2s ease-in-out infinite",
          boxShadow: "0 0 8px rgba(79, 195, 247, 0.4)",
        }}
      />
      Systematic FX Signals
    </span>
  );
}

/* ─── stat card ─── */
function StatCard({
  label,
  value,
  suffix,
  prefix,
  decimals,
  note,
  i,
}: {
  label: string;
  value: number;
  suffix?: string;
  prefix?: string;
  decimals?: number;
  note?: string;
  i: number;
}) {
  return (
    <motion.div
      variants={fadeUp}
      custom={i}
      style={{
        padding: "32px 24px",
        background: "rgba(255,255,255,0.015)",
        border: "1px solid rgba(79, 195, 247, 0.06)",
        borderRadius: 8,
        transition: "all 0.3s ease",
      }}
      whileHover={{
        borderColor: "rgba(79, 195, 247, 0.15)",
        background: "rgba(255,255,255,0.025)",
      }}
    >
      <div
        style={{
          fontSize: 10,
          fontFamily: "'DM Mono', monospace",
          color: "#4fc3f7",
          letterSpacing: "0.12em",
          textTransform: "uppercase",
          marginBottom: 16,
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontSize: 30,
          fontFamily: "'Syne', sans-serif",
          fontWeight: 700,
          color: "#e8eaf0",
          lineHeight: 1,
          marginBottom: note ? 12 : 0,
        }}
      >
        <Counter value={value} suffix={suffix} prefix={prefix} decimals={decimals ?? 2} />
      </div>
      {note && (
        <div
          style={{
            fontSize: 11,
            fontFamily: "'DM Mono', monospace",
            color: "#4b5563",
            lineHeight: 1.4,
          }}
        >
          {note}
        </div>
      )}
    </motion.div>
  );
}

/* ─── main page ─── */
export default function Home() {
  const [modalOpen, setModalOpen] = useState(false);
  const [modalSubject, setModalSubject] = useState("Schedule a Call");

  const openModal = (subject: string) => {
    setModalSubject(subject);
    setModalOpen(true);
  };

  return (
    <Layout onContact={() => openModal("Inquiry")}>
      <ContactModal
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
        subject={modalSubject}
      />
      {/* ══════ HERO ══════ */}
      <div
        style={{
          position: "relative",
          minHeight: "100vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          overflow: "hidden",
        }}
      >
        <GridBackground />

        {/* Radial glow */}
        <div
          style={{
            position: "absolute",
            top: "20%",
            left: "50%",
            transform: "translateX(-50%)",
            width: 800,
            height: 800,
            borderRadius: "50%",
            background: "radial-gradient(circle, rgba(79,195,247,0.04) 0%, transparent 70%)",
            pointerEvents: "none",
          }}
        />

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1.2 }}
          className="hero-container"
          style={{
            position: "relative",
            textAlign: "center" as const,
            maxWidth: 780,
            padding: "0 24px",
          }}
        >
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
            style={{ marginBottom: 32 }}
          >
            <LivePulse />
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.9, delay: 0.5 }}
            className="hero-title"
            style={{
              fontFamily: "'Syne', sans-serif",
              fontSize: 64,
              fontWeight: 700,
              color: "#e8eaf0",
              lineHeight: 1.1,
              letterSpacing: "-0.02em",
              marginBottom: 28,
            }}
          >
            Quantitative edge
            <br />
            <span style={{ color: "#4fc3f7" }}>for FX markets</span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.7 }}
            className="hero-subtitle"
            style={{
              fontFamily: "'DM Mono', monospace",
              fontSize: 14,
              color: "#6b7280",
              lineHeight: 1.8,
              maxWidth: 540,
              margin: "0 auto 48px",
            }}
          >
            LumenY delivers systematic, data-driven directional signals
            across major and cross FX pairs — built for institutional
            desks seeking independent alpha generation.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.9 }}
            className="hero-ctas"
            style={{ display: "flex", gap: 16, justifyContent: "center" }}
          >
            <a
              onClick={(e) => { e.preventDefault(); openModal("Schedule a Call"); }}
              href="#"
              style={{
                padding: "14px 32px",
                background: "rgba(79, 195, 247, 0.1)",
                border: "1px solid rgba(79, 195, 247, 0.3)",
                color: "#4fc3f7",
                fontSize: 12,
                fontFamily: "'DM Mono', monospace",
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                textDecoration: "none",
                borderRadius: 4,
                cursor: "pointer",
                transition: "all 0.25s ease",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = "rgba(79, 195, 247, 0.18)";
                e.currentTarget.style.borderColor = "rgba(79, 195, 247, 0.5)";
                e.currentTarget.style.transform = "translateY(-1px)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = "rgba(79, 195, 247, 0.1)";
                e.currentTarget.style.borderColor = "rgba(79, 195, 247, 0.3)";
                e.currentTarget.style.transform = "translateY(0)";
              }}
            >
              Schedule a Call
            </a>
            <a
              onClick={(e) => { e.preventDefault(); openModal("Inquiry"); }}
              href="#"
              style={{
                padding: "14px 32px",
                background: "transparent",
                border: "1px solid rgba(255,255,255,0.08)",
                color: "#a0a8b8",
                fontSize: 12,
                fontFamily: "'DM Mono', monospace",
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                textDecoration: "none",
                borderRadius: 4,
                cursor: "pointer",
                transition: "all 0.25s ease",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = "rgba(255,255,255,0.15)";
                e.currentTarget.style.color = "#e8eaf0";
                e.currentTarget.style.transform = "translateY(-1px)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = "rgba(255,255,255,0.08)";
                e.currentTarget.style.color = "#a0a8b8";
                e.currentTarget.style.transform = "translateY(0)";
              }}
            >
              Inquiries
            </a>
          </motion.div>

          {/* scroll hint */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.5, duration: 1 }}
            style={{
              position: "absolute",
              bottom: -100,
              left: "50%",
              transform: "translateX(-50%)",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 8,
            }}
          >
            <div
              style={{
                width: 1,
                height: 40,
                background: "linear-gradient(to bottom, rgba(79,195,247,0.2), transparent)",
              }}
            />
          </motion.div>
        </motion.div>
      </div>

      {/* ══════ APPROACH ══════ */}
      <Section id="approach">
        <motion.div variants={fadeUp} custom={0}>
          <div
            style={{
              fontSize: 10,
              fontFamily: "'DM Mono', monospace",
              color: "#4fc3f7",
              letterSpacing: "0.15em",
              textTransform: "uppercase",
              marginBottom: 16,
            }}
          >
            Our Approach
          </div>
          <h2
            style={{
              fontFamily: "'Syne', sans-serif",
              fontSize: 38,
              fontWeight: 700,
              color: "#e8eaf0",
              lineHeight: 1.2,
              marginBottom: 20,
              maxWidth: 500,
            }}
            className="section-heading"
          >
            Systematic intelligence,
            <br />
            not discretion
          </h2>
          <p
            style={{
              fontFamily: "'DM Mono', monospace",
              fontSize: 13,
              color: "#6b7280",
              lineHeight: 1.8,
              maxWidth: 560,
              marginBottom: 64,
            }}
          >
            Our model processes high-dimensional market data through a
            purely mathematical framework to generate directional probability
            signals across major and cross currency pairs. No discretionary
            overlay, no narrative bias — only rigorous quantitative analysis.
          </p>
        </motion.div>

        <div
          className="approach-grid"
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(3, 1fr)",
            gap: 24,
          }}
        >
          {[
            {
              title: "High-Dimensional",
              desc: "Each signal is derived from a dense feature space engineered to capture market dynamics invisible to conventional analysis.",
              icon: "◎",
            },
            {
              title: "FX Coverage",
              desc: "Active signal generation across major pairs and select crosses — EURUSD, GBPUSD, AUDUSD, USDJPY, EURGBP, and more.",
              icon: "⬡",
            },
            {
              title: "Mathematical Rigor",
              desc: "Built on a statistical learning framework — every signal is a probability, every decision is model-driven. No heuristics, no intuition.",
              icon: "◇",
            },
          ].map((item, i) => (
            <motion.div
              key={item.title}
              variants={fadeUp}
              custom={i + 1}
              style={{
                padding: "40px 32px",
                background: "rgba(255,255,255,0.015)",
                border: "1px solid rgba(255,255,255,0.04)",
                borderRadius: 8,
                transition: "all 0.3s ease",
              }}
              whileHover={{
                borderColor: "rgba(79, 195, 247, 0.1)",
                background: "rgba(255,255,255,0.025)",
              }}
            >
              <div
                style={{
                  fontSize: 24,
                  color: "#4fc3f7",
                  marginBottom: 20,
                  opacity: 0.7,
                }}
              >
                {item.icon}
              </div>
              <h3
                style={{
                  fontFamily: "'Syne', sans-serif",
                  fontSize: 16,
                  fontWeight: 600,
                  color: "#e8eaf0",
                  marginBottom: 12,
                }}
              >
                {item.title}
              </h3>
              <p
                style={{
                  fontFamily: "'DM Mono', monospace",
                  fontSize: 12,
                  color: "#6b7280",
                  lineHeight: 1.7,
                }}
              >
                {item.desc}
              </p>
            </motion.div>
          ))}
        </div>
      </Section>

      {/* ══════ PERFORMANCE ══════ */}
      <Section id="performance">
        <motion.div variants={fadeUp} custom={0}>
          <div
            style={{
              fontSize: 10,
              fontFamily: "'DM Mono', monospace",
              color: "#4fc3f7",
              letterSpacing: "0.15em",
              textTransform: "uppercase",
              marginBottom: 16,
            }}
          >
            Performance
          </div>
          <h2
            style={{
              fontFamily: "'Syne', sans-serif",
              fontSize: 38,
              fontWeight: 700,
              color: "#e8eaf0",
              lineHeight: 1.2,
              marginBottom: 12,
              maxWidth: 500,
            }}
            className="section-heading"
          >
            Results that speak
          </h2>
          <p
            style={{
              fontFamily: "'DM Mono', monospace",
              fontSize: 13,
              color: "#6b7280",
              lineHeight: 1.8,
              maxWidth: 560,
              marginBottom: 16,
            }}
          >
            Key metrics from a systematic trading test conducted from
            September 2025 to March 2026 across multiple FX pairs using
            LumenY signals.
          </p>
          <p
            style={{
              fontFamily: "'DM Mono', monospace",
              fontSize: 11,
              color: "#374151",
              lineHeight: 1.6,
              maxWidth: 560,
              marginBottom: 64,
            }}
          >
            Past performance is not indicative of future results. These figures
            reflect a controlled test environment, not live client returns.
          </p>
        </motion.div>

        <div
          className="stats-grid-3"
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(3, 1fr)",
            gap: 16,
            marginBottom: 16,
          }}
        >
          <StatCard label="Return" value={26.56} suffix="%" decimals={2} prefix="+" note="Over test period" i={1} />
          <StatCard label="Win Rate" value={65.1} suffix="%" decimals={1} note="446 / 685 trades" i={2} />
          <StatCard label="Sharpe Ratio" value={8.02} decimals={2} note="Annualized" i={3} />
        </div>
        <div
          className="stats-grid-2"
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(2, 1fr)",
            gap: 16,
            marginBottom: 32,
          }}
        >
          <StatCard label="Profit Factor" value={2.18} decimals={2} note="Gross profit / gross loss" i={4} />
          <StatCard label="Max Drawdown" value={1.55} suffix="%" decimals={2} note="Peak-to-trough" i={5} />
        </div>

        <div
          className="info-grid"
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(2, 1fr)",
            gap: 20,
          }}
        >
          <motion.div
            variants={fadeUp}
            custom={5}
            style={{
              padding: "32px",
              background: "rgba(255,255,255,0.015)",
              border: "1px solid rgba(255,255,255,0.04)",
              borderRadius: 8,
            }}
          >
            <div
              style={{
                fontSize: 10,
                fontFamily: "'DM Mono', monospace",
                color: "#4b5563",
                letterSpacing: "0.12em",
                textTransform: "uppercase",
                marginBottom: 20,
              }}
            >
              Test Information
            </div>
            <div className="info-inner-grid" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
              {[
                { k: "Period", v: "Sep 2025 – Mar 2026" },
                { k: "Pairs", v: "15 FX pairs" },
                { k: "Avg. trades / day", v: "3.4" },
                { k: "Total trades", v: "685" },
                { k: "Leverage", v: "50:1" },
              ].map(({ k, v }) => (
                <div key={k}>
                  <div
                    style={{
                      fontSize: 10,
                      fontFamily: "'DM Mono', monospace",
                      color: "#374151",
                      marginBottom: 4,
                      textTransform: "uppercase",
                      letterSpacing: "0.08em",
                    }}
                  >
                    {k}
                  </div>
                  <div
                    style={{
                      fontSize: 14,
                      fontFamily: "'DM Mono', monospace",
                      color: "#a0a8b8",
                    }}
                  >
                    {v}
                  </div>
                </div>
              ))}
            </div>
          </motion.div>

          <motion.div
            variants={fadeUp}
            custom={6}
            style={{
              padding: "32px",
              background: "rgba(255,255,255,0.015)",
              border: "1px solid rgba(255,255,255,0.04)",
              borderRadius: 8,
              display: "flex",
              flexDirection: "column",
              justifyContent: "center",
            }}
          >
            <div
              style={{
                fontSize: 10,
                fontFamily: "'DM Mono', monospace",
                color: "#4b5563",
                letterSpacing: "0.12em",
                textTransform: "uppercase",
                marginBottom: 20,
              }}
            >
              Signal Characteristics
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {[
                { label: "Avg. Win / Avg. Loss", value: "1.17x" },
                { label: "Consistency", value: "Positive across all test months" },
                { label: "Methodology", value: "Fully systematic, zero discretion" },
              ].map(({ label, value }) => (
                <div
                  key={label}
                  className="signal-row"
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    paddingBottom: 12,
                    borderBottom: "1px solid rgba(255,255,255,0.03)",
                  }}
                >
                  <span
                    style={{
                      fontSize: 12,
                      fontFamily: "'DM Mono', monospace",
                      color: "#6b7280",
                    }}
                  >
                    {label}
                  </span>
                  <span
                    style={{
                      fontSize: 12,
                      fontFamily: "'DM Mono', monospace",
                      color: "#a0a8b8",
                    }}
                  >
                    {value}
                  </span>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </Section>

      {/* ══════ CTA ══════ */}
      <section
        className="cta-section"
        style={{
          padding: "140px 48px",
          textAlign: "center" as const,
          position: "relative",
        }}
      >
        {/* glow */}
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            width: 600,
            height: 600,
            borderRadius: "50%",
            background: "radial-gradient(circle, rgba(79,195,247,0.03) 0%, transparent 70%)",
            pointerEvents: "none",
          }}
        />

        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-80px" }}
          style={{ position: "relative" }}
        >
          <motion.div
            variants={fadeUp}
            custom={0}
            style={{
              fontSize: 10,
              fontFamily: "'DM Mono', monospace",
              color: "#4fc3f7",
              letterSpacing: "0.15em",
              textTransform: "uppercase",
              marginBottom: 24,
            }}
          >
            Get Started
          </motion.div>

          <motion.h2
            variants={fadeUp}
            custom={1}
            style={{
              fontFamily: "'Syne', sans-serif",
              fontSize: 42,
              fontWeight: 700,
              color: "#e8eaf0",
              lineHeight: 1.2,
              marginBottom: 20,
            }}
            className="cta-heading"
          >
            Interested in our signals?
          </motion.h2>

          <motion.p
            variants={fadeUp}
            custom={2}
            style={{
              fontFamily: "'DM Mono', monospace",
              fontSize: 13,
              color: "#6b7280",
              lineHeight: 1.8,
              maxWidth: 480,
              margin: "0 auto 48px",
            }}
          >
            We work with institutional desks, funds, and systematic
            trading teams. Reach out to discuss signal integration
            and delivery.
          </motion.p>

          <motion.div
            variants={fadeUp}
            custom={3}
            className="cta-buttons"
            style={{ display: "flex", gap: 16, justifyContent: "center" }}
          >
            <a
              onClick={(e) => { e.preventDefault(); openModal("Schedule a Call"); }}
              href="#"
              style={{
                padding: "14px 36px",
                background: "rgba(79, 195, 247, 0.1)",
                border: "1px solid rgba(79, 195, 247, 0.3)",
                color: "#4fc3f7",
                fontSize: 12,
                fontFamily: "'DM Mono', monospace",
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                textDecoration: "none",
                borderRadius: 4,
                cursor: "pointer",
                transition: "all 0.25s ease",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = "rgba(79, 195, 247, 0.18)";
                e.currentTarget.style.borderColor = "rgba(79, 195, 247, 0.5)";
                e.currentTarget.style.transform = "translateY(-1px)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = "rgba(79, 195, 247, 0.1)";
                e.currentTarget.style.borderColor = "rgba(79, 195, 247, 0.3)";
                e.currentTarget.style.transform = "translateY(0)";
              }}
            >
              Schedule a Call
            </a>
            <a
              onClick={(e) => { e.preventDefault(); openModal("Inquiry"); }}
              href="#"
              style={{
                padding: "14px 36px",
                background: "transparent",
                border: "1px solid rgba(255,255,255,0.08)",
                color: "#a0a8b8",
                fontSize: 12,
                fontFamily: "'DM Mono', monospace",
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                textDecoration: "none",
                borderRadius: 4,
                cursor: "pointer",
                transition: "all 0.25s ease",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = "rgba(255,255,255,0.15)";
                e.currentTarget.style.color = "#e8eaf0";
                e.currentTarget.style.transform = "translateY(-1px)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = "rgba(255,255,255,0.08)";
                e.currentTarget.style.color = "#a0a8b8";
                e.currentTarget.style.transform = "translateY(0)";
              }}
            >
              Inquiries
            </a>
          </motion.div>
        </motion.div>
      </section>
    </Layout>
  );
}
