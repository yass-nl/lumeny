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

export default function TermsOfService() {
  return (
    <Layout>
      <div
        className="legal-page"
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
          Terms of Service
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

        <div style={sectionStyle}>
          <h2 style={headingStyle}>1. Overview</h2>
          <p style={textStyle}>
            These Terms of Service ("Terms") govern your access to and use of
            the LumenY website (lumen-y.com) and any signal services provided
            by LumenY ("we", "our", "us"). By accessing our website or using
            our services, you agree to be bound by these Terms.
          </p>
        </div>

        <div style={sectionStyle}>
          <h2 style={headingStyle}>2. Nature of Services</h2>
          <p style={textStyle}>
            LumenY provides quantitative directional signals and related
            informational content for foreign exchange (FX) markets. Our
            services are strictly informational in nature. We do not provide
            financial advice, investment advice, or portfolio management
            services. We do not manage, hold, or have access to client funds
            or trading accounts.
          </p>
          <p style={textStyle}>
            Our signals represent quantitative, model-generated outputs
            reflecting directional probabilities. They are not recommendations
            to buy, sell, or hold any financial instrument.
          </p>
        </div>

        <div style={sectionStyle}>
          <h2 style={headingStyle}>3. Client Responsibility</h2>
          <p style={textStyle}>
            You are solely responsible for any and all trading decisions made
            using LumenY signals. You acknowledge that trading foreign exchange
            carries significant risk, including the potential loss of your
            entire investment. You are responsible for conducting your own
            due diligence and for ensuring that any use of our signals
            complies with applicable laws and regulations in your jurisdiction.
          </p>
          <p style={textStyle}>
            You agree that LumenY bears no responsibility for any financial
            losses, damages, or other consequences arising from your use of
            our signals, whether directly or indirectly.
          </p>
        </div>

        <div style={sectionStyle}>
          <h2 style={headingStyle}>4. No Guarantee of Performance</h2>
          <p style={textStyle}>
            Past performance, including any test results or historical metrics
            presented on our website, is not indicative of future results. We
            make no guarantees, representations, or warranties regarding the
            accuracy, reliability, or profitability of our signals. Market
            conditions are inherently unpredictable and can change rapidly.
          </p>
        </div>

        <div style={sectionStyle}>
          <h2 style={headingStyle}>5. Intellectual Property</h2>
          <p style={textStyle}>
            All content on this website, including but not limited to text,
            graphics, logos, data, models, methodologies, and signal outputs,
            is the intellectual property of LumenY. You may not reproduce,
            distribute, modify, or create derivative works from any content
            without our prior written consent. Signal data provided to
            clients is licensed for the client's internal use only and may
            not be redistributed.
          </p>
        </div>

        <div style={sectionStyle}>
          <h2 style={headingStyle}>6. Confidentiality</h2>
          <p style={textStyle}>
            Any signal data, research, or proprietary information shared with
            clients is confidential. You agree not to disclose, share, or
            redistribute any such information to third parties without our
            prior written consent.
          </p>
        </div>

        <div style={sectionStyle}>
          <h2 style={headingStyle}>7. Limitation of Liability</h2>
          <p style={textStyle}>
            To the fullest extent permitted by applicable law, LumenY shall
            not be liable for any indirect, incidental, special, consequential,
            or punitive damages, including but not limited to loss of profits,
            data, or trading losses, arising out of or in connection with your
            use of our website or services.
          </p>
        </div>

        <div style={sectionStyle}>
          <h2 style={headingStyle}>8. Modifications</h2>
          <p style={textStyle}>
            We reserve the right to modify these Terms at any time. Changes
            will be posted on this page with an updated revision date.
            Continued use of our services following any changes constitutes
            acceptance of the updated Terms.
          </p>
        </div>

        <div style={sectionStyle}>
          <h2 style={headingStyle}>9. Governing Law</h2>
          <p style={textStyle}>
            These Terms shall be governed by and construed in accordance with
            applicable law. Any disputes arising under these Terms shall be
            resolved in accordance with the dispute resolution mechanisms
            agreed upon between the parties.
          </p>
        </div>

        <div style={sectionStyle}>
          <h2 style={headingStyle}>10. Contact</h2>
          <p style={textStyle}>
            For questions regarding these Terms, contact us at:{" "}
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
