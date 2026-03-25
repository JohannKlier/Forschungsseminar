"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import styles from "./page.module.css";

const pretrainedModel = "model:bike_hourly";

export default function Home() {
  const router = useRouter();
  const [hasAgreed, setHasAgreed] = useState(false);
  const [showAgreementWarning, setShowAgreementWarning] = useState(false);
  const studyHref = `/gam-lab?${new URLSearchParams({ model: pretrainedModel }).toString()}`;

  return (
    <div className={styles.page}>
      <main className={styles.main}>
        <section className={styles.hero}>
          <h1 className={styles.title}>Improve model performance by editing shape functions.</h1>
          <p className={styles.subtitle}>
            This study is designed for people who know the problem domain well, even if they do not have a machine
            learning background. Your job is to look at how the system currently behaves and correct patterns that do
            not match your real-world expectations.
          </p>
          <div className={styles.studyBrief}>
            <p className={styles.sectionLabel}>About this interface</p>
            <p className={styles.noteText}>
              This interface is related to GAM Changer. Instead of asking you to tune technical model settings, it lets
              you directly adjust simple curves that show how each input factor influences the prediction. You can use
              your domain knowledge to decide when one of these curves looks wrong or incomplete.
            </p>
            <p className={styles.sectionLabel}>Your task</p>
            <ul className={styles.studyList}>
              <li>Inspect the pretrained bike-sharing model and look for feature effects that do not fit what you know about the domain.</li>
              <li>Edit those curves so the model responds in a more sensible way.</li>
              <li>Use the feedback in the interface to see whether your changes improve the predictions.</li>
            </ul>
            <p className={styles.sectionLabel}>Before you start</p>
            <p className={styles.noteText}>
              You will begin with a model that is already trained. You do not need to understand the underlying machine
              learning method in detail. What matters here is whether the displayed relationships make sense from a
              domain perspective, and whether your edits help the model make better predictions.
            </p>
          </div>

          <label
            className={`${styles.agreement} ${showAgreementWarning && !hasAgreed ? styles.agreementWarning : ""}`}
          >
            <input
              className={styles.agreementCheckbox}
              type="checkbox"
              checked={hasAgreed}
              onChange={(event) => {
                setHasAgreed(event.target.checked);
                if (event.target.checked) {
                  setShowAgreementWarning(false);
                }
              }}
            />
            <span className={styles.agreementCopy}>
              I understand that my interactions in the study interface may be recorded for research purposes and that
              the collected data will be used to analyze how participants edit and evaluate models.
            </span>
          </label>

          <div className={styles.actions}>
            <button
              type="button"
              className={styles.primary}
              onClick={() => {
                if (hasAgreed) {
                  router.push(studyHref);
                  return;
                }
                setShowAgreementWarning(true);
              }}
            >
              Get started
            </button>
            <Link className={styles.secondary} href="/gam-lab/train">
              Full training interface
            </Link>
            <Link className={styles.secondary} href="/logs">
              Inspect user logs
            </Link>
          </div>
        </section>
      </main>
    </div>
  );
}
