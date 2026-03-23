// @ts-nocheck
import Layout from "../components/Layout";

const sectionStyle: React.CSSProperties = {
  marginBottom: 40,
};

const headingStyle: React.CSSProperties = {
  fontFamily: "'Syne', sans-serif",
  fontSize: 16,
  fontWeight: 600,
  color: "#e8eaf0",
  marginBottom: 16,
};

const textStyle: React.CSSProperties = {
  fontFamily: "'DM Mono', monospace",
  fontSize: 12,
  color: "#6b7280",
  lineHeight: 1.8,
  marginBottom: 12,
};

export default function RiskDisclosure() {
  return (
    <Layout>
      <div
        style={{
          maxWidth: 720,
          margin: "0 auto",
          padding: "140px 48px 120px",
        }}
      >
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
          Legal
        </div>
        <h1
          style={{
            fontFamily: "'Syne', sans-serif",
            fontSize: 36,
            fontWeight: 700,
            color: "#e8eaf0",
            marginBottom: 12,
          }}
        >
          Risk Disclosure
        </h1>
        <p
          style={{
            fontFamily: "'DM Mono', monospace",
            fontSize: 11,
            color: "#374151",
            marginBottom: 64,
          }}
        >
          Last updated: March 2026
        </p>

        <div
          style={{
            padding: "24px 28px",
            background: "rgba(79, 195, 247, 0.04)",
            border: "1px solid rgba(79, 195, 247, 0.1)",
            borderRadius: 8,
            marginBottom: 48,
          }}
        >
          <p
            style={{
              fontFamily: "'DM Mono', monospace",
              fontSize: 12,
              color: "#a0a8b8",
              lineHeight: 1.8,
              margin: 0,
            }}
          >
            Please read this Risk Disclosure statement carefully before using
            any services provided by LumenY. Trading foreign exchange involves
            significant risk and is not suitable for all participants.
          </p>
        </div>

        <div style={sectionStyle}>
          <h2 style={headingStyle}>1. Nature of Our Services</h2>
          <p style={textStyle}>
            LumenY provides quantitative, model-generated directional signals
            for foreign exchange (FX) markets. These signals are strictly
            informational and represent statistical probabilities derived from
            our proprietary models. They are not, and should not be construed
            as, financial advice, investment recommendations, or solicitations
            to trade.
          </p>
          <p style={textStyle}>
            LumenY does not manage funds, execute trades, or hold client assets.
            We provide information only. How that information is used is
            entirely at the discretion and responsibility of the recipient.
          </p>
        </div>

        <div style={sectionStyle}>
          <h2 style={headingStyle}>2. Risks of FX Trading</h2>
          <p style={textStyle}>
            Foreign exchange trading carries a high level of risk. The
            leveraged nature of FX trading means that small market movements
            can have a disproportionate impact on your capital. You may sustain
            losses in excess of your initial investment. You should not engage
            in FX trading unless you fully understand the risks involved and
            can afford to bear potential losses.
          </p>
        </div>

        <div style={sectionStyle}>
          <h2 style={headingStyle}>3. No Guarantee of Results</h2>
          <p style={textStyle}>
            Any performance metrics, test results, historical data, or
            statistical information presented on our website or in our
            communications are provided for informational purposes only.
            Past performance — whether from live trading, testing, or
            simulation — is not indicative of future results.
          </p>
          <p style={textStyle}>
            Market conditions change. Models that have performed well
            historically may not continue to do so. We make no representations
            or warranties regarding the accuracy, completeness, or reliability
            of our signals, nor do we guarantee any specific outcome from their
            use.
          </p>
        </div>

        <div style={sectionStyle}>
          <h2 style={headingStyle}>4. Client Responsibility</h2>
          <p style={textStyle}>
            By using LumenY signals, you acknowledge and accept that:
          </p>
          <ul style={{ ...textStyle, paddingLeft: 24 }}>
            <li style={{ marginBottom: 8 }}>
              You are solely responsible for all trading and investment
              decisions you make.
            </li>
            <li style={{ marginBottom: 8 }}>
              You have sufficient knowledge and experience to evaluate the
              risks of FX trading.
            </li>
            <li style={{ marginBottom: 8 }}>
              You will not rely solely on LumenY signals for trading decisions,
              and you will conduct your own analysis and due diligence.
            </li>
            <li style={{ marginBottom: 8 }}>
              LumenY is not responsible for any losses, damages, or
              consequences resulting from your use of our signals, whether
              directly or indirectly.
            </li>
            <li style={{ marginBottom: 8 }}>
              You are responsible for compliance with all applicable laws and
              regulations in your jurisdiction.
            </li>
          </ul>
        </div>

        <div style={sectionStyle}>
          <h2 style={headingStyle}>5. Model Limitations</h2>
          <p style={textStyle}>
            Quantitative models, including those used by LumenY, are subject to
            inherent limitations. They rely on historical data and statistical
            patterns that may not persist in the future. Models may be affected
            by unforeseen market events, regime changes, liquidity shifts, or
            other factors that fall outside the scope of historical analysis.
          </p>
        </div>

        <div style={sectionStyle}>
          <h2 style={headingStyle}>6. Independent Advice</h2>
          <p style={textStyle}>
            We recommend that you seek independent financial, legal, and tax
            advice before making any trading decisions. Our signals should be
            viewed as one input among many in your decision-making process, not
            as a standalone basis for action.
          </p>
        </div>

        <div style={sectionStyle}>
          <h2 style={headingStyle}>7. Contact</h2>
          <p style={textStyle}>
            If you have questions about this Risk Disclosure, contact us at:{" "}
            <a
              href="mailto:info@lumen-y.com"
              style={{ color: "#4fc3f7", textDecoration: "none" }}
            >
              info@lumen-y.com
            </a>
          </p>
        </div>
      </div>
    </Layout>
  );
}
