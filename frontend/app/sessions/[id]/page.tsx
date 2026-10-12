import { Suspense } from "react";

import SessionView from "@/components/SessionView";
import { Empty } from "@/components/ui";

async function SessionLoader({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <SessionView id={id} />;
}

export default function SessionPage({ params }: { params: Promise<{ id: string }> }) {
  return (
    <Suspense fallback={<Empty>Loading session...</Empty>}>
      <SessionLoader params={params} />
    </Suspense>
  );
}
