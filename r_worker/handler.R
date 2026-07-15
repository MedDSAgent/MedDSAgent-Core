# R execution handler.
#
# Implements the six methods the agent's RExecutorTool relies on:
#   execute, inject, get_state, save_state, load_state, reset_state
#
# State lives in a dedicated environment (not globalenv) so reset_state is a
# clean swap and save/load target exactly the user's variables.
#
# NOTE ON STDOUT: stdout is the protocol channel. Nothing here may print to it.
# All user output is captured with capture.output(); R's message()/warning()
# go to stderr, which is safely ignored.

EXCLUDED_VARS <- c("WORK_DIR", "UPLOADS_DIR", "OUTPUTS_DIR", "SCRIPTS_DIR", "INTERNAL_DIR")

# Blocked when MEDDS_CODE_GATE=true. Mirrors the Python RHandler's list.
BLOCKED_R_PATTERNS <- list(
  list(pattern = "\\bsystem\\s*\\(",           label = "system()"),
  list(pattern = "\\bsystem2\\s*\\(",          label = "system2()"),
  list(pattern = "\\bshell\\s*\\(",            label = "shell()"),
  list(pattern = "\\bshell\\.exec\\s*\\(",     label = "shell.exec()"),
  list(pattern = "\\bdownload\\.file\\s*\\(",  label = "download.file()"),
  list(pattern = "\\bsocketConnection\\s*\\(", label = "socketConnection()"),
  list(pattern = "\\burl\\s*\\(",              label = "url()")
)

RHandler <- function(work_dir) {
  self <- new.env(parent = emptyenv())
  self$work_dir <- normalizePath(work_dir, mustWork = FALSE)
  self$env <- new.env(parent = globalenv())

  setup_environment <- function() {
    assign("WORK_DIR",     self$work_dir,                          envir = self$env)
    assign("UPLOADS_DIR",  file.path(self$work_dir, "uploads"),    envir = self$env)
    assign("OUTPUTS_DIR",  file.path(self$work_dir, "outputs"),    envir = self$env)
    assign("SCRIPTS_DIR",  file.path(self$work_dir, "scripts"),    envir = self$env)
    assign("INTERNAL_DIR", file.path(self$work_dir, "internal"),   envir = self$env)
    if (dir.exists(self$work_dir)) setwd(self$work_dir)
  }

  code_gate_on <- function() {
    tolower(Sys.getenv("MEDDS_CODE_GATE", "false")) %in% c("true", "1", "yes")
  }

  check_code_safety <- function(code) {
    issues <- character(0)
    for (rule in BLOCKED_R_PATTERNS) {
      if (grepl(rule$pattern, code)) {
        issues <- c(issues, paste0("blocked call: ", rule$label))
      }
    }
    issues
  }

  # --- execute -------------------------------------------------------------
  # Evaluates each top-level expression, capturing printed output. Autoprints
  # visible results the way an interactive R session would.
  self$execute <- function(code) {
    if (code_gate_on()) {
      issues <- check_code_safety(code)
      if (length(issues) > 0) {
        return(paste0("[Blocked] Code contains restricted operations: ",
                      paste(issues, collapse = "; ")))
      }
    }

    if (dir.exists(self$work_dir)) setwd(self$work_dir)

    exprs <- NULL
    parse_err <- NULL
    tryCatch(
      exprs <- parse(text = code),
      error = function(e) parse_err <<- conditionMessage(e)
    )
    if (!is.null(parse_err)) return(paste0("[Error]\n", parse_err))

    err  <- NULL
    warns <- character(0)

    out <- utils::capture.output({
      for (ex in exprs) {
        res <- withCallingHandlers(
          tryCatch(
            withVisible(eval(ex, envir = self$env)),
            error = function(e) { err <<- conditionMessage(e); NULL }
          ),
          warning = function(w) {
            warns <<- c(warns, conditionMessage(w))
            invokeRestart("muffleWarning")
          }
        )
        if (!is.null(err)) break
        # Mimic the REPL: print only results that would autoprint.
        if (!is.null(res) && isTRUE(res$visible)) print(res$value)
      }
    })

    parts <- character(0)
    if (length(out) > 0) {
      joined <- paste(out, collapse = "\n")
      if (nzchar(trimws(joined))) parts <- c(parts, sub("\\s+$", "", joined))
    }
    if (length(warns) > 0) {
      parts <- c(parts, paste0("[stderr]\n", paste(warns, collapse = "\n")))
    }
    if (!is.null(err)) parts <- c(parts, paste0("[Error]\n", err))

    if (length(parts) == 0) "(No output)" else paste(parts, collapse = "\n")
  }

  # --- inject --------------------------------------------------------------
  # Silent evaluation for setup / DB connection code. Returns NULL on success
  # or an error string, matching the Python handler's contract.
  self$inject <- function(code) {
    err <- NULL
    tryCatch(
      utils::capture.output(eval(parse(text = code), envir = self$env)),
      error = function(e) err <<- conditionMessage(e)
    )
    err
  }

  self$reset_state <- function() {
    self$env <- new.env(parent = globalenv())
    setup_environment()
    invisible(NULL)
  }

  # --- save/load state -----------------------------------------------------
  # .RData via save()/load(), the same format the rpy2 handler used, so state
  # written by the old Python R worker still loads here.
  self$save_state <- function(path) {
    keys <- ls(self$env, all.names = FALSE)
    save(list = keys, file = path, envir = self$env)
    invisible(NULL)
  }

  self$load_state <- function(path) {
    if (file.exists(path) && file.info(path)$size > 0) {
      load(file = path, envir = self$env)
    }
    invisible(NULL)
  }

  # --- get_state -----------------------------------------------------------
  self$get_state <- function() {
    names_all <- ls(self$env, all.names = FALSE)
    names_all <- setdiff(names_all, EXCLUDED_VARS)
    lapply(names_all, function(nm) describe_var(nm, get(nm, envir = self$env)))
  }

  setup_environment()
  self
}

# --- variable description ---------------------------------------------------

capture_preview <- function(obj, max_chars = 2000) {
  txt <- tryCatch(
    paste(utils::capture.output(print(obj)), collapse = "\n"),
    error = function(e) paste0("<unprintable: ", conditionMessage(e), ">")
  )
  substr(txt, 1, max_chars)
}

is_ggplot_obj <- function(obj) inherits(obj, "ggplot")

# Renders a plot to a base64 <img>, matching the Python handler's output so the
# frontend renders it identically.
render_plot_b64 <- function(obj) {
  tmp <- tempfile(fileext = ".png")
  on.exit(unlink(tmp), add = TRUE)
  grDevices::png(tmp, width = 600, height = 400)
  ok <- tryCatch({ print(obj); TRUE }, error = function(e) FALSE, finally = grDevices::dev.off())
  if (!ok || !file.exists(tmp)) return(NULL)
  raw_bytes <- readBin(tmp, "raw", n = file.info(tmp)$size)
  b64 <- jsonlite::base64_enc(raw_bytes)
  paste0('<img src="data:image/png;base64,', b64, '" style="max-width:100%; height:auto;">')
}

html_escape <- function(x) {
  x <- gsub("&", "&amp;", x, fixed = TRUE)
  x <- gsub("<", "&lt;",  x, fixed = TRUE)
  x <- gsub(">", "&gt;",  x, fixed = TRUE)
  x
}

# Hand-rolled to match pandas' to_html(classes='df-table', border=0, index=False),
# which is what the Python handler produced and what the frontend styles.
# This is the one thing rpy2 was really buying us, and it is ~15 lines of R.
df_to_html <- function(df, max_rows = 500) {
  df <- utils::head(df, max_rows)
  cols <- names(df)
  header <- paste0("      <th>", html_escape(cols), "</th>", collapse = "\n")

  fmt_cell <- function(v) {
    s <- tryCatch(as.character(v), error = function(e) "<?>")
    s[is.na(s)] <- "NaN"
    html_escape(s)
  }
  col_strs <- lapply(df, fmt_cell)

  body <- ""
  if (nrow(df) > 0) {
    rows <- vapply(seq_len(nrow(df)), function(i) {
      cells <- vapply(col_strs, function(c) c[i], character(1))
      paste0("    <tr>\n", paste0("      <td>", cells, "</td>", collapse = "\n"), "\n    </tr>")
    }, character(1))
    body <- paste(rows, collapse = "\n")
  }

  paste0(
    '<table class="dataframe df-table">\n',
    "  <thead>\n    <tr>\n", header, "\n    </tr>\n  </thead>\n",
    "  <tbody>\n", body, "\n  </tbody>\n",
    "</table>"
  )
}

describe_var <- function(nm, obj) {
  r_class    <- tryCatch(class(obj)[1], error = function(e) "unknown")
  value_info <- ""
  preview    <- ""
  is_error   <- FALSE

  tryCatch({
    if (is_ggplot_obj(obj)) {
      r_class    <- "ggplot"
      value_info <- "Plot"
      img <- render_plot_b64(obj)
      preview <- if (is.null(img)) "Could not render plot" else img

    } else if (is.data.frame(obj)) {
      r_class    <- "data.frame"
      value_info <- sprintf("(%dx%d)", nrow(obj), ncol(obj))
      preview    <- tryCatch(df_to_html(obj), error = function(e) capture_preview(obj))

    } else if (is.matrix(obj)) {
      r_class    <- "matrix"
      d <- dim(obj)
      if (length(d) >= 2) value_info <- sprintf("(%dx%d)", d[1], d[2])
      preview <- capture_preview(obj)

    } else if (is.factor(obj)) {
      # Checked before is.atomic: factors are atomic but should report levels.
      r_class    <- "factor"
      value_info <- sprintf("(%d) [%d levels]", length(obj), nlevels(obj))
      preview    <- capture_preview(obj)

    } else if (is.function(obj)) {
      r_class <- "function"
      preview <- capture_preview(obj)

    } else if (is.list(obj)) {
      r_class    <- "list"
      value_info <- sprintf("(%d)", length(obj))
      preview    <- capture_preview(obj)

    } else if (is.atomic(obj)) {
      r_class <- if (is.integer(obj)) "integer"
      else if (is.numeric(obj))   "numeric"
      else if (is.character(obj)) "character"
      else if (is.logical(obj))   "logical"
      else r_class
      value_info <- sprintf("(%d)", length(obj))
      if (length(obj) == 1) {
        val <- tryCatch(as.character(obj)[1], error = function(e) "")
        if (is.na(val)) val <- "NA"
        value_info <- if (nchar(val) > 20) paste0(substr(val, 1, 20), "...") else val
      }
      preview <- capture_preview(obj)

    } else {
      value_info <- r_class
      preview    <- capture_preview(obj)
    }
  }, error = function(e) {
    is_error   <<- TRUE
    value_info <<- "Error"
    preview    <<- conditionMessage(e)
  })

  list(
    name     = nm,
    type     = r_class,
    value    = value_info,
    preview  = preview,
    is_error = is_error
  )
}
