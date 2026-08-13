const slider =
    document.getElementById(
        "epoch-slider"
    );

const image =
    document.getElementById(
        "sample-image"
    );

const epochNumber =
    document.getElementById(
        "epoch-number"
    );

const epochMetrics =
    document.getElementById(
        "epoch-metrics"
    );


function formatValue(value) {

    if (
        value === undefined ||
        value === null ||
        value === ""
    ) {
        return "";
    }

    if (
        typeof value !== "number"
    ) {
        return String(value);
    }

    if (
        Number.isInteger(value)
    ) {
        return String(value);
    }

    return value.toPrecision(4);
}


function updateViewer() {

    if (!samples.length) {
        return;
    }

    const index =
        Number(slider.value);

    const sample =
        samples[index];

    image.src =
        "images/" +
        sample.filename;

    image.alt =
        "Generated samples at epoch " +
        sample.epoch;

    epochNumber.textContent =
        sample.epoch;


    /*
     * A single epoch may have multiple
     * metrics.csv rows.
     *
     * Merge them into one row, with later
     * non-empty values taking precedence.
     */

    const epochRows =
        metrics.filter(
            row =>
                Number(row.epoch) ===
                Number(sample.epoch)
        );


    const row = {};


    for (
        const candidate of epochRows
    ) {

        for (
            const [key, value]
            of Object.entries(candidate)
        ) {

            if (
                value !== undefined &&
                value !== null &&
                value !== ""
            ) {
                row[key] = value;
            }

        }

    }


    const fields = [
        ["train_loss", "train"],
        ["val_loss", "val"],
        ["val_bpd", "BPD"],
        ["val_perplexity", "perplexity"]
    ];


    epochMetrics.textContent =
        fields
            .filter(
                ([key]) =>
                    row[key] !== undefined
            )
            .map(
                ([key, label]) =>
                    label +
                    " " +
                    formatValue(row[key])
            )
            .join(" · ");
}


slider.addEventListener(
    "input",
    updateViewer
);


updateViewer();


/*
 * Metrics table
 */

if (metrics.length) {

    const preferredColumns = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_bpd",
        "val_perplexity"
    ];


    const columns =
        preferredColumns.filter(
            column =>
                metrics.some(
                    row =>
                        column in row
                )
        );


    const head =
        document.getElementById(
            "metrics-head"
        );

    const body =
        document.getElementById(
            "metrics-body"
        );


    columns.forEach(
        column => {

            const th =
                document.createElement(
                    "th"
                );

            th.textContent =
                column
                    .replaceAll("_", " ");

            head.appendChild(th);

        }
    );


    /*
     * Merge the raw CSV rows by epoch.
     */

    const epochMap =
        new Map();


    for (const sourceRow of metrics) {

        const epoch =
            sourceRow.epoch;

        if (
            epoch === undefined ||
            epoch === null ||
            epoch === ""
        ) {
            continue;
        }


        const key =
            String(epoch);


        if (!epochMap.has(key)) {
            epochMap.set(key, {});
        }


        const target =
            epochMap.get(key);


        for (
            const [column, value]
            of Object.entries(sourceRow)
        ) {

            if (
                value !== undefined &&
                value !== null &&
                value !== ""
            ) {
                target[column] = value;
            }

        }

    }


    const epochRows =
        Array.from(
            epochMap.values()
        );


    /*
     * Don't make a huge table if there
     * are hundreds of epochs.
     */

    const maxRows = 20;

    const step =
        Math.max(
            1,
            Math.ceil(
                epochRows.length /
                maxRows
            )
        );


    epochRows.forEach(
        (row, index) => {

            if (
                index % step !== 0 &&
                index !==
                    epochRows.length - 1
            ) {
                return;
            }


            const tr =
                document.createElement(
                    "tr"
                );


            columns.forEach(
                column => {

                    const td =
                        document.createElement(
                            "td"
                        );

                    td.textContent =
                        formatValue(
                            row[column]
                        );

                    tr.appendChild(td);

                }
            );


            body.appendChild(tr);

        }
    );
}