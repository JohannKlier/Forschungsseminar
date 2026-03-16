"use client";

import Link from "next/link";
import styles from "./page.module.css";

const pretrainedModel = "model:bike_hourly";

export default function Home() {
  const studyHref = `/gam-lab?${new URLSearchParams({ model: pretrainedModel }).toString()}`;

  return (
    <div className={styles.page}>
      <main className={styles.main}>
        <section className={styles.hero}>
          <span className={styles.eyebrow}>User Study</span>
          <h1 className={styles.title}>Improve model performance by editing shape functions.</h1>
          <p className={styles.subtitle}>
            In this study, you will inspect a pretrained interpretable model, use your own knowledge of the problem
            domain, and revise feature shape functions to produce better predictions.
          </p>

          <div className={styles.studyPanel}>
            <div className={styles.studyLead}>
              <p className={styles.sectionLabel}>What you will do</p>
              <ul className={styles.studyList}>
                <li>Inspect feature effects in a pretrained bike-sharing model.</li>
                <li>Use your knowledge to identify shape functions that should behave differently.</li>
                <li>Edit those effects and evaluate whether the model predictions improve.</li>
              </ul>
            </div>

            <div className={styles.studyNote}>
              <p className={styles.sectionLabel}>Before you start</p>
              <p className={styles.noteText}>
                The goal is to improve predictive performance through informed edits, not to retrain from scratch. The
                session opens directly with a pretrained model so you can begin immediately.
              </p>
            </div>
          </div>

          <div className={styles.actions}>
            <Link className={styles.primary} href={studyHref}>
              Start with pretrained model
            </Link>
            <Link className={styles.primary} href="/logs">
              Inspect user logs
            </Link>
          </div>
        </section>

        <section className={styles.featureGrid}>
          <article className={styles.card}>
            <p className={styles.cardKicker}>Goal</p>
            <h2 className={styles.cardTitle}>Improve the model</h2>
            <p className={styles.cardText}>
              Use the interface to turn your knowledge about the task into edits that improve predictive behavior.
            </p>
          </article>
          <article className={styles.card}>
            <p className={styles.cardKicker}>Task</p>
            <h2 className={styles.cardTitle}>Revise feature effects</h2>
            <p className={styles.cardText}>
              Adjust shapes that appear unrealistic, incomplete, or unhelpful for accurate predictions.
            </p>
          </article>
          <article className={styles.card}>
            <p className={styles.cardKicker}>Output</p>
            <h2 className={styles.cardTitle}>Deliver a stronger version</h2>
            <p className={styles.cardText}>
              Aim for a revised model that performs better and remains easy to explain through its shape functions.
            </p>
          </article>
        </section>
      </main>
    </div>
  );
}
