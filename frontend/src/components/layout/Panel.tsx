import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

/** Glass panel used for each of the three dashboard columns. */
export function Panel({
  title,
  icon,
  children,
  className,
  bodyClassName,
  actions,
}: {
  title?: string;
  icon?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
  actions?: ReactNode;
}) {
  return (
    <section
      className={cn(
        "glass flex min-h-0 flex-col overflow-hidden rounded-2xl",
        className,
      )}
    >
      {title ? (
        <div className="flex shrink-0 items-center gap-2 border-b border-border/50 px-3.5 py-2.5">
          {icon ? <span className="text-primary">{icon}</span> : null}
          <h2 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            {title}
          </h2>
          {actions ? <div className="ml-auto flex items-center gap-1">{actions}</div> : null}
        </div>
      ) : null}
      <div className={cn("min-h-0 flex-1 overflow-y-auto scrollbar-thin p-3", bodyClassName)}>
        {children}
      </div>
    </section>
  );
}
