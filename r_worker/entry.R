#!/usr/bin/env Rscript
#
# R subprocess worker entry point.
#
# Launched by the TS WorkerProcess (src/workers/). Implements the same JSON-line
# contract as python_worker/entry.py — see docs/worker-protocol.md.
#
# Launch pattern:
#     Rscript r_worker/entry.R --work_dir /path/to/session
#
# Protocol:
#   - One JSON object per line, both directions.
#   - On startup: one {"status":"ok","data":<ready_info>} line, or
#     {"status":"error","error":"..."} followed by exit 1.
#   - Command:  {"method": "...", "params": {...}}
#   - Response: {"status":"ok","data":{...}} | {"status":"error","error":"..."}
#   - "shutdown" exits cleanly.
#
# STDOUT DISCIPLINE: stdout carries protocol lines only. User code output is
# captured via capture.output() in handler.R; R sends message()/warning()/
# package startup banners to stderr, which never touches this channel.

local({
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  script_dir <- if (length(file_arg) > 0) {
    dirname(normalizePath(sub("^--file=", "", file_arg[1])))
  } else {
    getwd()
  }
  source(file.path(script_dir, "handler.R"), local = FALSE)
})

# --- protocol io -------------------------------------------------------------

.proto_out <- stdout()

write_line <- function(obj) {
  # auto_unbox: scalars must serialize as JSON scalars, not 1-element arrays.
  # null="null" so an absent error field survives the round trip.
  line <- jsonlite::toJSON(obj, auto_unbox = TRUE, null = "null", na = "null", digits = NA)
  cat(line, "\n", sep = "", file = .proto_out)
  flush(.proto_out)
}

write_error <- function(msg) {
  write_line(list(status = jsonlite::unbox("error"), error = jsonlite::unbox(as.character(msg))))
}

write_ok <- function(data) {
  if (is.null(data)) data <- structure(list(), names = character(0))
  write_line(list(status = jsonlite::unbox("ok"), data = data))
}

# --- arg parsing (mirrors entry.py: --key value pairs) ----------------------

parse_kwargs <- function(argv) {
  kwargs <- list()
  i <- 1
  while (i <= length(argv)) {
    tok <- argv[i]
    if (startsWith(tok, "--")) {
      key <- substring(tok, 3)
      val <- if (i + 1 <= length(argv)) argv[i + 1] else TRUE
      kwargs[[key]] <- val
      i <- i + 2
    } else {
      i <- i + 1
    }
  }
  kwargs
}

OPTIONAL_LIBS <- c("ggplot2", "dplyr", "tidyr", "data.table", "survival")

get_ready_info <- function() {
  present <- Filter(function(p) requireNamespace(p, quietly = TRUE), OPTIONAL_LIBS)
  list(
    r_version = jsonlite::unbox(R.version.string),
    # I(): force a JSON array even for 0 or 1 element. Without it an empty result
    # serializes as null and a single one as a bare string, and the parent expects
    # an array in both cases.
    available_libs = I(as.character(present))
  )
}

# --- dispatch ---------------------------------------------------------------

dispatch <- function(handler, method, params) {
  if (method == "execute") {
    code <- if (is.null(params$code)) "" else params$code
    return(list(output = jsonlite::unbox(handler$execute(code))))

  } else if (method == "inject") {
    code <- if (is.null(params$code)) "" else params$code
    err <- handler$inject(code)
    return(list(error = if (is.null(err)) NULL else jsonlite::unbox(err)))

  } else if (method == "get_state") {
    vars <- handler$get_state()
    # Force a JSON array even when there are 0 or 1 variables.
    return(list(variables = if (length(vars) == 0) I(list()) else vars))

  } else if (method == "save_state") {
    if (is.null(params$path)) stop("save_state requires 'path'")
    handler$save_state(params$path)
    return(structure(list(), names = character(0)))

  } else if (method == "load_state") {
    if (is.null(params$path)) stop("load_state requires 'path'")
    handler$load_state(params$path)
    return(structure(list(), names = character(0)))

  } else if (method == "reset_state") {
    handler$reset_state()
    return(structure(list(), names = character(0)))
  }

  stop(sprintf("RHandler: unknown method '%s'", method))
}

# --- main -------------------------------------------------------------------

main <- function() {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    # Cannot use write_error(): it needs jsonlite. Hand-build the one line.
    cat('{"status":"error","error":"The R package \'jsonlite\' is required by the MedDSAgent R worker but is not installed. Install it with: install.packages(\'jsonlite\')"}\n')
    quit(status = 1)
  }

  argv <- commandArgs(trailingOnly = TRUE)
  kwargs <- parse_kwargs(argv)

  handler <- tryCatch(
    {
      if (is.null(kwargs$work_dir)) stop("Missing required argument --work_dir")
      RHandler(work_dir = kwargs$work_dir)
    },
    error = function(e) {
      write_error(paste0("Failed to initialize R handler: ", conditionMessage(e)))
      quit(status = 1)
    }
  )

  ready <- tryCatch(get_ready_info(), error = function(e) list())
  write_ok(ready)

  con <- file("stdin", open = "r")
  repeat {
    raw <- readLines(con, n = 1, warn = FALSE)
    if (length(raw) == 0) break          # EOF — parent closed the pipe
    raw <- trimws(raw)
    if (!nzchar(raw)) next

    cmd <- tryCatch(
      jsonlite::fromJSON(raw, simplifyVector = FALSE),
      error = function(e) NULL
    )
    if (is.null(cmd)) {
      write_error(paste0("Invalid JSON: ", raw))
      next
    }

    method <- if (is.null(cmd$method)) "" else cmd$method
    params <- if (is.null(cmd$params)) list() else cmd$params

    if (identical(method, "shutdown")) {
      write_ok(structure(list(), names = character(0)))
      break
    }

    result <- tryCatch(
      dispatch(handler, method, params),
      error = function(e) structure(conditionMessage(e), class = "worker_error")
    )

    if (inherits(result, "worker_error")) {
      write_error(as.character(result))
    } else {
      write_ok(result)
    }
  }
}

main()
